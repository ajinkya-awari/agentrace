"""Kaggle-only, six-call maximum diagnostic verifying the GPT-OSS JSON-mode fix.

Root cause established by gpt_oss_request_matrix.py (2026-09-05, 8 calls):
plain-text requests succeed on both GPT-OSS models; every JSON Object Mode
request fails with `json_validate_failed` regardless of max_tokens vs
max_completion_tokens or direct SDK vs ChatGroq. GPT-OSS models on Groq are
reasoning models whose hidden reasoning tokens count against the completion
budget, so JSON Object Mode can be truncated before a valid JSON object is
emitted. This script isolates whether `reasoning_effort` and/or a larger
token budget resolves it, without touching the production contract until
one candidate passes cleanly on both models.
"""

import importlib.util
import os
from typing import Any, Callable

from agentrace.evidence import EVIDENCE_DIR, write_stage_evidence
from notebooks.response_contract_probe import _safe_error_metadata, classify_probe_error

DEPENDENCY_INSTALL_COMMAND = "python -m pip install -r requirements.txt"
REQUIRED_PACKAGES = ("groq", "langchain_core", "langchain_groq")
GPT_OSS_MODELS = ("openai/gpt-oss-120b", "openai/gpt-oss-20b")
JSON_DIRECT_PROMPT = [
    {"role": "system", "content": "Return a valid JSON object only, exactly {\"answer\": \"A\"}."},
    {"role": "user", "content": "Synthetic request-shape diagnostic. Return JSON only."},
]
FIX_PROFILES = (
    {"name": "reasoning_low_same_budget_32", "runner": "reasoning_low_same_budget"},
    {"name": "reasoning_low_larger_budget_256", "runner": "reasoning_low_larger_budget"},
    {"name": "no_reasoning_larger_budget_256", "runner": "no_reasoning_larger_budget"},
)


def ensure_matrix_dependencies(package_available: Callable[[str], bool] | None = None) -> None:
    available = package_available or (lambda package: importlib.util.find_spec(package) is not None)
    if any(not available(package) for package in REQUIRED_PACKAGES):
        raise RuntimeError(f"Missing notebook dependency. Install with: {DEPENDENCY_INSTALL_COMMAND}")


def _http_status(error: Exception) -> int | None:
    status = getattr(error, "status_code", None)
    response = getattr(error, "response", None)
    status = status if isinstance(status, int) else getattr(response, "status_code", None)
    return status if isinstance(status, int) else None


def sanitize_fix_failure(*, model: str, profile: str, error: Exception) -> dict[str, Any]:
    """Return exactly the approved sanitized record without retaining exception text."""
    provider_type, code, parameter = _safe_error_metadata(error)
    return {
        "model": model,
        "profile": profile,
        "status": "fail",
        "http_status": _http_status(error),
        "provider_type": provider_type,
        "code": code,
        "parameter": parameter,
        "category": classify_probe_error(error),
    }


def _success_record(model: str, profile: str) -> dict[str, Any]:
    return {
        "model": model,
        "profile": profile,
        "status": "pass",
        "http_status": None,
        "provider_type": None,
        "code": None,
        "parameter": None,
        "category": "ok",
    }


def _direct_client():
    from groq import Groq

    return Groq()


def _run_reasoning_low_same_budget(client: Any, model: str) -> None:
    client.chat.completions.create(
        model=model,
        messages=JSON_DIRECT_PROMPT,
        response_format={"type": "json_object"},
        max_completion_tokens=32,
        reasoning_effort="low",
        temperature=0,
    )


def _run_reasoning_low_larger_budget(client: Any, model: str) -> None:
    client.chat.completions.create(
        model=model,
        messages=JSON_DIRECT_PROMPT,
        response_format={"type": "json_object"},
        max_completion_tokens=256,
        reasoning_effort="low",
        temperature=0,
    )


def _run_no_reasoning_larger_budget(client: Any, model: str) -> None:
    client.chat.completions.create(
        model=model,
        messages=JSON_DIRECT_PROMPT,
        response_format={"type": "json_object"},
        max_completion_tokens=256,
        temperature=0,
    )


def run_fix_verification() -> list[dict[str, Any]]:
    """Execute exactly one request for each of three candidate fixes on both GPT-OSS models."""
    client = _direct_client()
    runners = {
        "reasoning_low_same_budget": lambda model: _run_reasoning_low_same_budget(client, model),
        "reasoning_low_larger_budget": lambda model: _run_reasoning_low_larger_budget(client, model),
        "no_reasoning_larger_budget": lambda model: _run_no_reasoning_larger_budget(client, model),
    }
    records = []
    for model in GPT_OSS_MODELS:
        for profile in FIX_PROFILES:
            try:
                runners[profile["runner"]](model)
            except Exception as error:
                records.append(sanitize_fix_failure(model=model, profile=profile["name"], error=error))
            else:
                records.append(_success_record(model, profile["name"]))
    return records


def main() -> None:
    ensure_matrix_dependencies()
    if os.getenv("AGENTRACE_NOTEBOOK_RUNTIME") != "1" or os.getenv("AGENTRACE_ALLOW_LIVE") != "1":
        raise RuntimeError("GPT-OSS fix verification is notebook-gated and requires explicit live approval")
    records = run_fix_verification()
    passed = sum(record["status"] == "pass" for record in records)
    path = write_stage_evidence(
        stage="gpt_oss_fix_verification",
        status="pass" if passed == len(records) else "fail",
        test_counts={"calls": len(records), "passed": passed, "failed": len(records) - passed},
        failure_summary="GPT-OSS fix verification contains failures" if passed != len(records) else None,
        matrix_diagnostics=records,
        output_dir=EVIDENCE_DIR,
    )
    if passed != len(records):
        raise RuntimeError(f"GPT-OSS fix verification completed with failures. Evidence: {path}")
    print(f"GPT-OSS fix verification passed. Evidence: {path}")


if __name__ == "__main__":
    main()
