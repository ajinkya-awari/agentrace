"""Kaggle-only, eight-call maximum diagnostic for GPT-OSS request incompatibilities."""

import importlib.util
import os
from typing import Any, Callable

from agentrace.evidence import EVIDENCE_DIR, write_stage_evidence
from notebooks.response_contract_probe import _safe_error_metadata, classify_probe_error

DEPENDENCY_INSTALL_COMMAND = "python -m pip install -r requirements.txt"
REQUIRED_PACKAGES = ("groq", "langchain_core", "langchain_groq")
GPT_OSS_MODELS = ("openai/gpt-oss-120b", "openai/gpt-oss-20b")
JSON_PROMPT = [
    ("system", "Return a valid JSON object only, exactly {\"answer\": \"A\"}."),
    ("human", "Synthetic request-shape diagnostic. Return JSON only."),
]
PLAIN_PROMPT = [{"role": "user", "content": "Synthetic request-shape diagnostic. Reply with exactly A."}]
JSON_DIRECT_PROMPT = [
    {"role": "system", "content": "Return a valid JSON object only, exactly {\"answer\": \"A\"}."},
    {"role": "user", "content": "Synthetic request-shape diagnostic. Return JSON only."},
]
REQUEST_PROFILES = (
    {"name": "direct_plain_max_completion_tokens_16", "runner": "direct_plain"},
    {"name": "direct_json_max_completion_tokens_32", "runner": "direct_json_completion"},
    {"name": "direct_json_max_tokens_32", "runner": "direct_json_tokens"},
    {"name": "chatgroq_json_max_completion_tokens_32", "runner": "chatgroq_json_completion"},
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


def sanitize_matrix_failure(*, model: str, profile: str, error: Exception) -> dict[str, Any]:
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


def _run_direct_plain(client: Any, model: str) -> None:
    client.chat.completions.create(model=model, messages=PLAIN_PROMPT, max_completion_tokens=16, temperature=0)


def _run_direct_json_completion(client: Any, model: str) -> None:
    client.chat.completions.create(
        model=model,
        messages=JSON_DIRECT_PROMPT,
        response_format={"type": "json_object"},
        max_completion_tokens=32,
        temperature=0,
    )


def _run_direct_json_tokens(client: Any, model: str) -> None:
    client.chat.completions.create(
        model=model,
        messages=JSON_DIRECT_PROMPT,
        response_format={"type": "json_object"},
        max_tokens=32,
        temperature=0,
    )


def _run_chatgroq_json_completion(model: str) -> None:
    from langchain_groq import ChatGroq

    llm = ChatGroq(
        model=model,
        temperature=0,
        model_kwargs={"response_format": {"type": "json_object"}, "max_completion_tokens": 32},
    )
    llm.invoke(JSON_PROMPT)


def run_matrix() -> list[dict[str, Any]]:
    """Execute exactly one request for each of four profiles and two GPT-OSS models."""
    client = _direct_client()
    direct_runners = {
        "direct_plain": lambda model: _run_direct_plain(client, model),
        "direct_json_completion": lambda model: _run_direct_json_completion(client, model),
        "direct_json_tokens": lambda model: _run_direct_json_tokens(client, model),
        "chatgroq_json_completion": _run_chatgroq_json_completion,
    }
    records = []
    for model in GPT_OSS_MODELS:
        for profile in REQUEST_PROFILES:
            try:
                direct_runners[profile["runner"]](model)
            except Exception as error:
                records.append(sanitize_matrix_failure(model=model, profile=profile["name"], error=error))
            else:
                records.append(_success_record(model, profile["name"]))
    return records


def main() -> None:
    ensure_matrix_dependencies()
    if os.getenv("AGENTRACE_NOTEBOOK_RUNTIME") != "1" or os.getenv("AGENTRACE_ALLOW_LIVE") != "1":
        raise RuntimeError("GPT-OSS request matrix is notebook-gated and requires explicit live approval")
    records = run_matrix()
    passed = sum(record["status"] == "pass" for record in records)
    path = write_stage_evidence(
        stage="gpt_oss_request_matrix",
        status="pass" if passed == len(records) else "fail",
        test_counts={"calls": len(records), "passed": passed, "failed": len(records) - passed},
        failure_summary="GPT-OSS request matrix contains failures" if passed != len(records) else None,
        matrix_diagnostics=records,
        output_dir=EVIDENCE_DIR,
    )
    if passed != len(records):
        raise RuntimeError(f"GPT-OSS request matrix completed with failures. Evidence: {path}")
    print(f"GPT-OSS request matrix passed. Evidence: {path}")


if __name__ == "__main__":
    main()
