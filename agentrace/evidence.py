"""Sanitized, dated evidence helpers for notebook-gated AgentTrace stages."""

from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
from typing import Any

EVIDENCE_DIR = Path("docs/evidence")
ALLOWED_STAGES = {
    "synthetic_validation",
    "response_contract_probe",
    "gpt_oss_request_matrix",
    "gpt_oss_fix_verification",
    "tracer_smoke",
    "mini_gate",
}
MATRIX_DIAGNOSTIC_STAGES = {"gpt_oss_request_matrix", "gpt_oss_fix_verification"}
ALLOWED_STATUSES = {"pass", "fail"}
TIMESTAMP_RE = re.compile(r"^\d{8}T\d{6}Z$")
BASELINE_DIAGNOSTIC_FIELDS = {
    "total_calls",
    "parsable_answers",
    "correct_answers",
    "incorrect_answers",
    "unparseable_answers",
    "provider_errors",
    "parse_empty_content",
    "parse_invalid_json",
    "parse_invalid_label",
    "parse_unsupported_content_shape",
    "provider_error_timeout",
    "provider_error_rate_limited",
    "provider_error_authentication",
    "provider_error_model_access",
    "provider_error_provider_http",
    "provider_error_provider_unknown",
}
PROBE_DIAGNOSTIC_FIELDS = {"status", "category", "calls"}
PROBE_STATUSES = {"pass", "fail"}
PROBE_CATEGORIES = {
    "ok",
    "missing_dependency",
    "authentication",
    "rate_limit",
    "timeout",
    "bad_request",
    "unsupported_parameter",
    "incompatible_parameter_combination",
    "response_format_rejected",
    "reasoning_configuration_rejected",
    "invalid_request",
    "unknown_bad_request",
    "model_unavailable",
    "parse_failure",
    "unknown",
}
CONTRACT_FIELDS = {"response_contract", "cache_version", "model_ids", "request_profiles"}
MATRIX_DIAGNOSTIC_FIELDS = {"model", "profile", "status", "http_status", "provider_type", "code", "parameter", "category"}


def _evidence_dir(output_dir: Path | None = None) -> Path:
    return output_dir or Path(os.getenv("AGENTRACE_EVIDENCE_DIR", EVIDENCE_DIR))


def source_identifier() -> str:
    """Return only a commit identifier and dirty marker, never repository content."""
    try:
        head = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return f"commit:{head}" + ("+dirty" if dirty else "")
    except (OSError, subprocess.SubprocessError):
        return "working-tree:unavailable"


def environment_summary() -> dict[str, Any]:
    packages = {}
    for name in ("langgraph", "langchain-core", "langchain-groq", "pytest"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not-installed"
    return {"platform": platform.platform(), "packages": packages}


def _failure_summary(value: str | None) -> str | None:
    return None if not value else "failure details withheld; inspect the notebook console"


def _validated_baseline_diagnostics(value: dict[str, dict[str, int]]) -> dict[str, dict[str, int]]:
    """Allow only fixed numeric counters in mini-gate evidence."""
    if not isinstance(value, dict):
        raise ValueError("baseline diagnostic payload must be a mapping")
    validated = {}
    for model, counts in value.items():
        if not isinstance(model, str) or not isinstance(counts, dict):
            raise ValueError("baseline diagnostic payload must contain model count mappings")
        if set(counts) - BASELINE_DIAGNOSTIC_FIELDS:
            raise ValueError("baseline diagnostic contains an unsafe field")
        if any(not isinstance(count, int) or count < 0 for count in counts.values()):
            raise ValueError("baseline diagnostic counts must be non-negative integers")
        validated[model] = dict(counts)
    return validated


def _validated_probe_diagnostics(value: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Allow probe status/category/count fields only; never raw provider payloads."""
    if not isinstance(value, dict):
        raise ValueError("probe diagnostic payload must be a mapping")
    validated = {}
    for model, result in value.items():
        if not isinstance(model, str) or not isinstance(result, dict) or set(result) != PROBE_DIAGNOSTIC_FIELDS:
            raise ValueError("probe diagnostic payload must contain bounded result mappings")
        if result["status"] not in PROBE_STATUSES or result["category"] not in PROBE_CATEGORIES:
            raise ValueError("probe diagnostic contains an unsafe category")
        if not isinstance(result["calls"], int) or result["calls"] < 0:
            raise ValueError("probe diagnostic calls must be a non-negative integer")
        validated[model] = dict(result)
    return validated


def _validated_contract(value: dict[str, Any]) -> dict[str, Any]:
    """Record only active source/model identifiers needed to interpret a probe."""
    if not isinstance(value, dict) or set(value) != CONTRACT_FIELDS:
        raise ValueError("contract payload must contain only the approved identifiers")
    if not isinstance(value["response_contract"], str) or not isinstance(value["cache_version"], str):
        raise ValueError("contract identifiers must be strings")
    model_ids = value["model_ids"]
    if not isinstance(model_ids, dict) or any(not isinstance(key, str) or not isinstance(model_id, str) for key, model_id in model_ids.items()):
        raise ValueError("contract model IDs must be a string mapping")
    request_profiles = value["request_profiles"]
    if not isinstance(request_profiles, dict) or any(not isinstance(key, str) or not isinstance(profile, str) for key, profile in request_profiles.items()):
        raise ValueError("contract request profiles must be a string mapping")
    return {
        "response_contract": value["response_contract"],
        "cache_version": value["cache_version"],
        "model_ids": dict(model_ids),
        "request_profiles": dict(request_profiles),
    }


def _validated_matrix_diagnostics(value: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Allow only fixed request-matrix metadata; never provider text or payloads."""
    if not isinstance(value, list):
        raise ValueError("matrix diagnostic payload must be a list")
    validated = []
    for record in value:
        if not isinstance(record, dict) or set(record) != MATRIX_DIAGNOSTIC_FIELDS:
            raise ValueError("matrix diagnostic payload must contain bounded records")
        if record["status"] not in PROBE_STATUSES or record["category"] not in PROBE_CATEGORIES:
            raise ValueError("matrix diagnostic contains an unsafe status or category")
        if record["http_status"] is not None and (not isinstance(record["http_status"], int) or record["http_status"] < 0):
            raise ValueError("matrix HTTP status must be a non-negative integer or null")
        if any(record[field] is not None and not isinstance(record[field], str) for field in ("model", "profile", "provider_type", "code", "parameter")):
            raise ValueError("matrix diagnostic fields must be strings or null")
        validated.append(dict(record))
    return validated


def write_stage_evidence(
    *,
    stage: str,
    status: str,
    test_counts: dict[str, int],
    failure_summary: str | None = None,
    baseline_diagnostics: dict[str, dict[str, int]] | None = None,
    probe_diagnostics: dict[str, dict[str, Any]] | None = None,
    matrix_diagnostics: list[dict[str, Any]] | None = None,
    contract: dict[str, Any] | None = None,
    output_dir: Path | None = None,
) -> Path:
    """Write a bounded JSON record without prompts, traces, secrets, or raw outputs."""
    if stage not in ALLOWED_STAGES:
        raise ValueError(f"Unknown evidence stage: {stage}")
    if status not in ALLOWED_STATUSES:
        raise ValueError(f"Unknown evidence status: {status}")
    if any(not isinstance(value, int) or value < 0 for value in test_counts.values()):
        raise ValueError("Evidence counts must be non-negative integers")

    now = datetime.now(timezone.utc)
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    path = _evidence_dir(output_dir) / f"runtime-{stamp}-{stage}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage": stage,
        "timestamp_utc": now.isoformat(),
        "source_identifier": source_identifier(),
        "python_version": platform.python_version(),
        "environment": environment_summary(),
        "status": status,
        "test_counts": dict(test_counts),
        "failure_summary": _failure_summary(failure_summary),
    }
    if baseline_diagnostics is not None:
        payload["baseline_diagnostics"] = _validated_baseline_diagnostics(baseline_diagnostics)
    if probe_diagnostics is not None:
        if stage != "response_contract_probe":
            raise ValueError("probe diagnostics are valid only for the response-contract probe")
        payload["probe_diagnostics"] = _validated_probe_diagnostics(probe_diagnostics)
    if contract is not None:
        if stage != "response_contract_probe":
            raise ValueError("contract identifiers are valid only for the response-contract probe")
        payload["contract"] = _validated_contract(contract)
    if matrix_diagnostics is not None:
        if stage not in MATRIX_DIAGNOSTIC_STAGES:
            raise ValueError("matrix diagnostics are valid only for GPT-OSS matrix/fix-verification stages")
        payload["matrix_diagnostics"] = _validated_matrix_diagnostics(matrix_diagnostics)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def has_valid_tracer_evidence(output_dir: Path | None = None) -> bool:
    """Accept only a dated, successful smoke record with the required event counts."""
    for path in sorted(_evidence_dir(output_dir).glob("runtime-*-tracer_smoke.json"), reverse=True):
        try:
            parts = path.name.split("-")
            if len(parts) != 3 or not TIMESTAMP_RE.fullmatch(parts[1]):
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("stage") != "tracer_smoke" or payload.get("status") != "pass":
                continue
            datetime.fromisoformat(str(payload["timestamp_utc"]).replace("Z", "+00:00"))
            counts = payload.get("test_counts", {})
            if int(counts.get("events", 0)) >= 5 and int(counts.get("llm_start_events", 0)) >= 1:
                return True
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
    return False
