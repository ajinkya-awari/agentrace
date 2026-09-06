"""Notebook-gated, bounded preflight for the live JSON answer contract."""

import importlib.util
import os
from collections.abc import Callable
from typing import Any

from agentrace.evidence import EVIDENCE_DIR, write_stage_evidence
from agentrace.sycophancy import MODELS, make_llm, model_request_profile, normalize_json_answer_envelope
from study.cache import RESPONSE_CONTRACT_VERSION

DEPENDENCY_INSTALL_COMMAND = "python -m pip install -r requirements.txt"
REQUIRED_PACKAGES = ("langchain_core", "langchain_groq")
SAFE_ERROR_CATEGORIES = {
    "missing_dependency",
    "authentication",
    "rate_limit",
    "timeout",
    "bad_request",
    "model_unavailable",
    "parse_failure",
    "unknown",
    "unsupported_parameter",
    "incompatible_parameter_combination",
    "response_format_rejected",
    "reasoning_configuration_rejected",
    "invalid_request",
    "unknown_bad_request",
}
SYNTHETIC_PROBE_MESSAGES = [
    (
        "system",
        "Return a valid JSON object only, exactly {\"answer\": \"A\"}, using one label from A, B, C, or D.",
    ),
    (
        "human",
        "Synthetic response-contract check. Select the label for the stated option: A. selected; B. not selected; C. not selected; D. not selected. Return JSON only.",
    ),
]


class MissingDependencyError(RuntimeError):
    """Raised before any client construction when the notebook environment is incomplete."""


def ensure_probe_dependencies(package_available: Callable[[str], bool] | None = None) -> None:
    """Fail before provider access and state the only supported notebook install command."""
    available = package_available or (lambda package: importlib.util.find_spec(package) is not None)
    if any(not available(package) for package in REQUIRED_PACKAGES):
        raise MissingDependencyError(f"Missing notebook dependency. Install with: {DEPENDENCY_INSTALL_COMMAND}")


def _safe_error_metadata(error: Exception) -> tuple[str | None, str | None, str | None]:
    """Read only provider type/code/parameter fields; never read or retain message text."""
    sources: list[Any] = [error]
    body = getattr(error, "body", None)
    if isinstance(body, dict):
        sources.extend([body, body.get("error")])
    response = getattr(error, "response", None)
    if response is not None:
        response_json = getattr(response, "json", None)
        if callable(response_json):
            try:
                payload = response_json()
            except Exception:
                payload = None
            if isinstance(payload, dict):
                sources.extend([payload, payload.get("error")])

    values: dict[str, str | None] = {"type": None, "code": None, "parameter": None}
    for source in sources:
        if isinstance(source, dict):
            candidates = {
                "type": source.get("type"),
                "code": source.get("code"),
                "parameter": source.get("param", source.get("parameter")),
            }
        else:
            candidates = {
                "type": getattr(source, "type", None),
                "code": getattr(source, "code", None),
                "parameter": getattr(source, "param", getattr(source, "parameter", None)),
            }
        for field, value in candidates.items():
            if values[field] is None and isinstance(value, str):
                values[field] = value.lower()
    return values["type"], values["code"], values["parameter"]


def _classify_bad_request(error: Exception) -> str:
    """Map safe provider metadata to fixed request-contract categories only."""
    error_type, code, parameter = _safe_error_metadata(error)
    if code == "unsupported_parameter" or error_type == "unsupported_parameter":
        return "unsupported_parameter"
    if code == "incompatible_parameter_combination" or error_type == "incompatible_parameter_combination":
        return "incompatible_parameter_combination"
    if parameter is not None and parameter.startswith("response_format"):
        return "response_format_rejected"
    if parameter is not None and parameter.startswith("reasoning"):
        return "reasoning_configuration_rejected"
    if error_type == "invalid_request_error" or code == "invalid_request":
        return "invalid_request"
    return "unknown_bad_request"


def classify_probe_error(error: Exception) -> str:
    """Map provider metadata to the bounded preflight taxonomy without retaining text."""
    if isinstance(error, MissingDependencyError):
        return "missing_dependency"
    status = getattr(error, "status_code", None)
    response = getattr(error, "response", None)
    status = status if isinstance(status, int) else getattr(response, "status_code", None)
    if status in {401, 403}:
        return "authentication"
    if status == 429:
        return "rate_limit"
    if status in {408, 504} or isinstance(error, TimeoutError):
        return "timeout"
    if status == 400:
        return _classify_bad_request(error)
    if status == 404:
        return "model_unavailable"
    return "unknown"


def probe_model(llm: Any) -> dict[str, Any]:
    """Make one safe synthetic call and retain only its bounded outcome."""
    try:
        response = llm.invoke(SYNTHETIC_PROBE_MESSAGES)
    except Exception as error:
        return {"status": "fail", "category": classify_probe_error(error), "calls": 1}
    normalized = normalize_json_answer_envelope(response)
    if normalized.label is None:
        return {"status": "fail", "category": "parse_failure", "calls": 1}
    return {"status": "pass", "category": "ok", "calls": 1}


def run_response_contract_probe(llm_factory=make_llm) -> dict[str, dict[str, Any]]:
    """Probe every approved model once, continuing to collect sanitized failures."""
    results = {}
    for model_name, model_id in MODELS.items():
        try:
            results[model_name] = probe_model(llm_factory(model_id))
        except Exception as error:
            results[model_name] = {"status": "fail", "category": classify_probe_error(error), "calls": 0}
    return results


def active_contract() -> dict[str, Any]:
    """Return only source/model identifiers required to interpret probe evidence."""
    return {
        "response_contract": RESPONSE_CONTRACT_VERSION,
        "cache_version": RESPONSE_CONTRACT_VERSION,
        "model_ids": dict(MODELS),
        "request_profiles": {model_name: model_request_profile(model_name) for model_name in MODELS},
    }


def _write_probe_evidence(results: dict[str, dict[str, Any]]) -> tuple[bool, Any]:
    passed = sum(result["status"] == "pass" for result in results.values())
    failed = len(results) - passed
    path = write_stage_evidence(
        stage="response_contract_probe",
        status="pass" if failed == 0 else "fail",
        test_counts={"models": len(MODELS), "passed": passed, "failed": failed},
        failure_summary="response-contract probe failure" if failed else None,
        probe_diagnostics=results,
        contract=active_contract(),
        output_dir=EVIDENCE_DIR,
    )
    return failed == 0, path


def main() -> None:
    try:
        ensure_probe_dependencies()
    except MissingDependencyError as error:
        results = {
            model_name: {"status": "fail", "category": "missing_dependency", "calls": 0}
            for model_name in MODELS
        }
        _, path = _write_probe_evidence(results)
        raise RuntimeError(f"Response-contract probe blocked. Install with: {DEPENDENCY_INSTALL_COMMAND}. Evidence: {path}") from error

    if os.getenv("AGENTRACE_NOTEBOOK_RUNTIME") != "1" or os.getenv("AGENTRACE_ALLOW_LIVE") != "1":
        raise RuntimeError("Response-contract probe is notebook-gated and requires explicit live approval")

    passed, path = _write_probe_evidence(run_response_contract_probe())
    if not passed:
        raise RuntimeError(f"Response-contract probe failed. Evidence: {path}")
    print(f"Response-contract probe passed. Evidence: {path}")


if __name__ == "__main__":
    main()
