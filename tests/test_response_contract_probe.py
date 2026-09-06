import json

import pytest

from agentrace.evidence import write_stage_evidence
from agentrace.sycophancy import MODELS
from notebooks import response_contract_probe as probe
from study.cache import RESPONSE_CONTRACT_VERSION, cache_key
from study.run_study import MINI_GATE_QUESTION_COUNT, dry_run_plan


def test_probe_calls_each_configured_model_once_and_records_only_bounded_outcomes():
    created = []
    invoked = []

    class FakeLLM:
        def __init__(self, model_id):
            self.model_id = model_id

        def invoke(self, messages):
            invoked.append((self.model_id, messages))
            return '{"answer": "A"}'

    results = probe.run_response_contract_probe(
        llm_factory=lambda model_id: created.append(model_id) or FakeLLM(model_id)
    )

    assert created == list(MODELS.values())
    assert [model_id for model_id, _ in invoked] == list(MODELS.values())
    assert results == {
        name: {"status": "pass", "category": "ok", "calls": 1}
        for name in MODELS
    }
    assert all("Synthetic" in messages[1][1] for _, messages in invoked)


@pytest.mark.parametrize(
    ("error", "category"),
    [
        (probe.MissingDependencyError(), "missing_dependency"),
        (type("Auth", (RuntimeError,), {"status_code": 401})(), "authentication"),
        (type("Rate", (RuntimeError,), {"status_code": 429})(), "rate_limit"),
        (TimeoutError(), "timeout"),
        (type("Request", (RuntimeError,), {"status_code": 400})(), "unknown_bad_request"),
        (type("Model", (RuntimeError,), {"status_code": 404})(), "model_unavailable"),
        (RuntimeError("details withheld"), "unknown"),
    ],
)
def test_probe_error_classifier_uses_the_safe_preflight_categories(error, category):
    assert probe.classify_probe_error(error) == category


@pytest.mark.parametrize(
    ("metadata", "category"),
    [
        ({"type": "invalid_request_error", "code": "unsupported_parameter", "param": "temperature"}, "unsupported_parameter"),
        ({"type": "invalid_request_error", "code": "incompatible_parameter_combination", "param": "response_format"}, "incompatible_parameter_combination"),
        ({"type": "invalid_request_error", "code": "invalid_parameter", "param": "response_format.type"}, "response_format_rejected"),
        ({"type": "invalid_request_error", "code": "invalid_parameter", "param": "reasoning_effort"}, "reasoning_configuration_rejected"),
        ({"type": "invalid_request_error", "code": "invalid_parameter", "param": "max_tokens"}, "invalid_request"),
        ({}, "unknown_bad_request"),
    ],
)
def test_probe_classifies_bad_request_using_only_sanitized_provider_metadata(metadata, category):
    class FakeBadRequest(RuntimeError):
        status_code = 400
        body = {"error": {**metadata, "message": "gsk_secret never retained"}}

    assert probe.classify_probe_error(FakeBadRequest()) == category


def test_probe_bad_request_result_does_not_retain_provider_error_metadata_or_message():
    class FakeBadRequest(RuntimeError):
        status_code = 400
        body = {"error": {"type": "invalid_request_error", "code": "unsupported_parameter", "param": "x", "message": "gsk_secret never retained"}}

    class FakeLLM:
        def invoke(self, _messages):
            raise FakeBadRequest()

    result = probe.probe_model(FakeLLM())

    assert result == {"status": "fail", "category": "unsupported_parameter", "calls": 1}
    assert "gsk_secret" not in repr(result)


def test_probe_reports_parse_failure_without_retaining_provider_payload():
    class FakeLLM:
        def invoke(self, _messages):
            return '{"answer": "not-a-label", "secret": "gsk_should_not_escape"}'

    result = probe.probe_model(FakeLLM())

    assert result == {"status": "fail", "category": "parse_failure", "calls": 1}
    assert "gsk_should_not_escape" not in repr(result)


def test_dependency_preflight_fails_with_the_exact_notebook_install_command():
    with pytest.raises(probe.MissingDependencyError, match="python -m pip install -r requirements.txt"):
        probe.ensure_probe_dependencies(package_available=lambda _name: False)


def test_probe_evidence_is_sanitized_and_identifies_the_active_contract(tmp_path):
    results = {
        name: {"status": "pass", "category": "ok", "calls": 1}
        for name in MODELS
    }
    path = write_stage_evidence(
        stage="response_contract_probe",
        status="pass",
        test_counts={"models": 3, "passed": 3, "failed": 0},
        probe_diagnostics=results,
        contract=probe.active_contract(),
        output_dir=tmp_path,
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert path.name.endswith("-response_contract_probe.json")
    assert payload["probe_diagnostics"] == results
    assert payload["contract"] == {
        "cache_version": RESPONSE_CONTRACT_VERSION,
        "model_ids": MODELS,
        "request_profiles": {
            "qwen-3.6-27b": "json_object_qwen_hidden_reasoning",
            "gpt-oss-120b": "json_object_gpt_oss_reasoning_low",
            "gpt-oss-20b": "json_object_gpt_oss_reasoning_low",
        },
        "response_contract": "json-answer-v1",
    }
    serialized = path.read_text(encoding="utf-8").lower()
    assert "gsk_" not in serialized
    assert "raw_payload" not in serialized
    assert "secret" not in serialized


def test_probe_contract_does_not_change_benchmark_eligibility_accounting_or_cache_versioning():
    plan = dry_run_plan(50)
    assert plan["mini_gate_calls"] == MINI_GATE_QUESTION_COUNT * 3 * 5
    assert plan["full_gate_calls"] == 750
    assert cache_key("model", "question", "vector").startswith(f"{RESPONSE_CONTRACT_VERSION}|")
