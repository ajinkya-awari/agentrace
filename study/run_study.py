"""Offline dry-run and notebook-gated benchmark orchestration."""

import argparse
import json
from typing import Any, Iterable

from agentrace.sycophancy import ATTACK_TEMPLATES, MODELS, make_llm, run_attack, wilson_ci
from study.cache import cache_key, load_cache, save_result


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
    raise RuntimeError("Live benchmark execution is notebook-gated and requires explicit approval")


if __name__ == "__main__":
    main()
