"""Explicitly invoked MedQA adapter and baseline-pass filter."""

from typing import Any

from agentrace.sycophancy import invoke_mcq

DATASET_ID = "GBaker/MedQA-USMLE-4-options"
OPTION_LABELS = ("A", "B", "C", "D")


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    correct = str(row["answer_idx"]).strip().upper()
    assert correct in OPTION_LABELS
    options = row["options"]
    assert set(options) == set(OPTION_LABELS)
    return {**row, "answer_idx": correct, "question_id": str(row.get("id", row.get("question_id", "unknown")))}


def load_baseline_questions(llms: dict[str, Any], n: int = 50) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("datasets is required only in the notebook runtime") from exc
    dataset = load_dataset(DATASET_ID, split="test")
    target = n
    passing: list[dict[str, Any]] = []
    for scanned, raw_row in enumerate(dataset, start=1):
        row = normalize_row(raw_row)
        if all(invoke_mcq(llm, row["question"], row["options"]) == row["answer_idx"] for llm in llms.values()):
            passing.append(row)
        print(f"Baseline scan: {len(passing)}/{target} passing, {scanned}/500 scanned")
        if len(passing) >= target or scanned >= 500:
            break
    if len(passing) < target and target > 30:
        passing = passing[:30]
    return passing
