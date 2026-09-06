"""Explicitly invoked MedQA adapter and baseline-pass filter."""

from typing import Any

from agentrace.sycophancy import (
    AnswerResult,
    PARSE_ERROR_CATEGORIES,
    PROVIDER_ERROR_CATEGORIES,
    classify_provider_error,
    invoke_mcq,
    normalize_answer_response,
)

DATASET_ID = "GBaker/MedQA-USMLE-4-options"
OPTION_LABELS = ("A", "B", "C", "D")


class BaselineEligibilityError(RuntimeError):
    def __init__(self, diagnostics: dict[str, dict[str, int]]):
        super().__init__("No sufficient all-model baseline-passing questions")
        self.diagnostics = diagnostics


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    correct = str(row["answer_idx"]).strip().upper()
    assert correct in OPTION_LABELS
    options = row["options"]
    assert set(options) == set(OPTION_LABELS)
    return {**row, "answer_idx": correct, "question_id": str(row.get("id", row.get("question_id", "unknown")))}


def baseline_filter_rows(rows, llms: dict[str, Any], n: int, invoke=invoke_mcq):
    diagnostics = {
        name: {
            "total_calls": 0,
            "parsable_answers": 0,
            "correct_answers": 0,
            "incorrect_answers": 0,
            "unparseable_answers": 0,
            "provider_errors": 0,
            **{f"parse_{category}": 0 for category in PARSE_ERROR_CATEGORIES},
            **{f"provider_error_{category}": 0 for category in PROVIDER_ERROR_CATEGORIES},
        }
        for name in llms
    }
    # Check GPT-OSS models before Qwen: GPT-OSS baseline pass rates are far lower (~2-3%)
    # than Qwen's, so failing fast on GPT-OSS first skips the other two calls for the vast
    # majority of candidates instead of always paying for all three every time.
    ordered_llms = sorted(llms.items(), key=lambda item: "gpt-oss" not in item[0])
    passing = []
    for scanned, raw_row in enumerate(rows, start=1):
        row = normalize_row(raw_row)
        eligible = True
        for name, llm in ordered_llms:
            counts = diagnostics[name]
            counts["total_calls"] += 1
            try:
                result = invoke(llm, row["question"], row["options"])
            except Exception as error:
                result = AnswerResult(None, classify_provider_error(error))
            if not isinstance(result, AnswerResult):
                result = normalize_answer_response(result)
            if result.category in PROVIDER_ERROR_CATEGORIES:
                counts["provider_errors"] += 1
                counts[f"provider_error_{result.category}"] += 1
                eligible = False
                break
            if result.label is None:
                counts["unparseable_answers"] += 1
                counts[f"parse_{result.category}"] += 1
                eligible = False
                break
            counts["parsable_answers"] += 1
            if result.label == row["answer_idx"]:
                counts["correct_answers"] += 1
            else:
                counts["incorrect_answers"] += 1
                eligible = False
                break
        if eligible:
            passing.append(row)
        print(f"Baseline scan: {len(passing)}/{n} passing, {scanned}/500 scanned")
        if len(passing) >= n or scanned >= 500:
            break
    return passing, diagnostics


def load_baseline_questions(llms: dict[str, Any], n: int = 50) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("datasets is required only in the notebook runtime") from exc
    dataset = load_dataset(DATASET_ID, split="test")
    target = n
    passing, diagnostics = baseline_filter_rows(dataset, llms, target)
    if len(passing) < target and target > 30:
        passing = passing[:30]
    if len(passing) < target:
        raise BaselineEligibilityError(diagnostics)
    return passing
