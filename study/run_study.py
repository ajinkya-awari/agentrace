"""Offline dry-run and notebook-gated benchmark orchestration."""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterable

from agentrace.sycophancy import ATTACK_TEMPLATES, MODELS, make_llm, run_attack, wilson_ci
from study.cache import cache_key, load_cache, save_result

RESULTS_DIR = Path("study/results")


def dry_run_plan(max_questions: int = 50) -> dict[str, Any]:
    return {
        "models": MODELS,
        "vectors": list(ATTACK_TEMPLATES),
        "questions": max_questions,
        "mini_gate_calls": 3 * len(MODELS) * len(ATTACK_TEMPLATES),
        "full_gate_calls": max_questions * len(MODELS) * len(ATTACK_TEMPLATES),
        "runtime": "notebook_only",
        "live_api": False,
    }


def aggregate(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in records:
        if record.get("attacked_answer") is not None:
            grouped.setdefault((record["model"], record["vector"]), []).append(record)
    rows = []
    for (model, vector), values in sorted(grouped.items()):
        n = len(values)
        flipped = sum(int(value["sycophancy"]) for value in values)
        ci_lo, ci_hi = wilson_ci(flipped, n)
        rows.append({"model": model, "vector": vector, "n_flipped": flipped, "n": n, "rate": round(100 * flipped / n, 1), "ci_lo": ci_lo, "ci_hi": ci_hi})
    return rows


def run_mini_gate(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Validate the 45-call gate before a full run; caller must be notebook-gated."""
    rows = aggregate(records)
    rates = [row["rate"] for row in rows]
    if any(rate < 0.0 or rate > 100.0 for rate in rates):
        raise ValueError("Mini-gate produced an impossible rate")
    if rates and all(rate == 0.0 for rate in rates):
        raise RuntimeError("Mini-gate diagnostic: verify wrong_claim label and text injection")
    if rates and all(rate == 100.0 for rate in rates):
        raise RuntimeError("Mini-gate diagnostic: verify answer parsing and scoring")
    return rows


def write_outputs(records: list[dict[str, Any]], table: list[dict[str, Any]]) -> None:
    """Write local evidence only when explicitly called from the notebook runtime."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "sycophancy_results.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    (RESULTS_DIR / "sycophancy_table.json").write_text(json.dumps(table, indent=2), encoding="utf-8")

    import matplotlib.pyplot as plt
    import seaborn as sns

    models = list(MODELS)
    vectors = list(ATTACK_TEMPLATES)
    values = {
        (row["model"], row["vector"]): row["rate"]
        for row in table
    }
    matrix = [[values.get((model, vector), 0.0) for vector in vectors] for model in models]
    figure, axis = plt.subplots(figsize=(11, 4))
    sns.heatmap(matrix, annot=True, fmt=".1f", vmin=0, vmax=100, cmap="mako", xticklabels=vectors, yticklabels=models, ax=axis)
    axis.set_xlabel("Attack vector")
    axis.set_ylabel("Model")
    figure.tight_layout()
    figure.savefig(RESULTS_DIR / "sycophancy_heatmap.png", dpi=160)
    plt.close(figure)


def log_wandb(table: list[dict[str, Any]]) -> None:
    """Upload exactly 15 aggregate rows only when the W&B gate is explicit."""
    if os.getenv("AGENTRACE_ALLOW_WANDB") != "1":
        return
    import wandb

    with wandb.init(project="agentrace-benchmark") as run:
        for row in table:
            run.log({"sycophancy_rate": row["rate"], "ci_lo": row["ci_lo"], "ci_hi": row["ci_hi"], "model": row["model"], "vector": row["vector"]})
        run.log({"sycophancy_heatmap": wandb.Image(str(RESULTS_DIR / "sycophancy_heatmap.png"))})
        artifact = wandb.Artifact("agentrace-study-results", type="benchmark")
        for name in ("sycophancy_results.json", "sycophancy_table.json", "sycophancy_heatmap.png", "nist_audit_report.json"):
            path = RESULTS_DIR / name
            if path.exists():
                artifact.add_file(str(path))
        run.log_artifact(artifact)


def run_benchmark(questions: list[dict[str, Any]], resume: bool = False) -> list[dict[str, Any]]:
    cache = load_cache() if resume else {}
    records = []
    for model_name, model_id in MODELS.items():
        llm = make_llm(model_id)
        for row in questions:
            question_id = str(row.get("question_id", row.get("id", "unknown")))
            for vector in ATTACK_TEMPLATES:
                key = cache_key(model_name, question_id, vector)
                if key in cache:
                    records.append(cache[key])
                    continue
                record = run_attack(llm, row, vector)
                record.update({"model": model_name, "question_id": question_id, "key": key})
                save_result(record)
                records.append(record)
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-questions", type=int, default=50)
    args = parser.parse_args()
    if args.dry_run:
        print(json.dumps(dry_run_plan(args.max_questions), indent=2))
        return
    if os.getenv("AGENTRACE_NOTEBOOK_RUNTIME") != "1" or os.getenv("AGENTRACE_ALLOW_LIVE") != "1":
        raise RuntimeError("Live benchmark execution is notebook-gated and requires explicit approval")

    from study.dataset import load_baseline_questions

    llms = {name: make_llm(model_id) for name, model_id in MODELS.items()}
    questions = load_baseline_questions(llms, n=args.max_questions)
    mini_records = run_benchmark(questions[:3], resume=args.resume)
    run_mini_gate(mini_records)
    records = run_benchmark(questions, resume=args.resume)
    table = aggregate(records)
    write_outputs(records, table)
    log_wandb(table)


if __name__ == "__main__":
    main()
