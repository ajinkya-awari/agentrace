import json
from types import SimpleNamespace

import pytest

from agentrace import evidence
from examples import medical_agent_audit
from notebooks import runtime_validation


def test_stage_evidence_uses_a_dated_sanitized_filename_and_payload(tmp_path):
    path = evidence.write_stage_evidence(
        stage="synthetic_validation",
        status="pass",
        test_counts={"passed": 7, "failed": 0, "errors": 0},
        failure_summary="gsk_secret prompt patient-id 123",
        output_dir=tmp_path,
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert path.name.startswith("runtime-")
    assert path.name.endswith("-synthetic_validation.json")
    assert set(payload) == {
        "stage",
        "timestamp_utc",
        "source_identifier",
        "python_version",
        "environment",
        "status",
        "test_counts",
        "failure_summary",
    }
    assert payload["failure_summary"] == "failure details withheld; inspect the notebook console"
    assert "gsk_secret" not in path.read_text(encoding="utf-8")
    assert evidence.has_valid_tracer_evidence(tmp_path) is False


def test_valid_passing_tracer_evidence_satisfies_the_gate(tmp_path):
    evidence.write_stage_evidence(
        stage="tracer_smoke",
        status="pass",
        test_counts={"events": 5, "llm_start_events": 1},
        output_dir=tmp_path,
    )

    assert evidence.has_valid_tracer_evidence(tmp_path) is True


def test_synthetic_validation_writes_evidence_without_running_a_provider(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTRACE_NOTEBOOK_RUNTIME", "1")
    monkeypatch.setenv("AGENTRACE_EVIDENCE_DIR", str(tmp_path))
    monkeypatch.setenv("AGENTRACE_ALLOW_LIVE", "1")

    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout="12 passed in 0.10s", stderr="")

    runtime_validation.run(command_runner=fake_run)

    payload = json.loads(next(tmp_path.glob("*-synthetic_validation.json")).read_text(encoding="utf-8"))
    assert calls == [[runtime_validation.sys.executable, "-m", "pytest", "-q"]]
    assert payload["status"] == "pass"
    assert payload["test_counts"] == {"passed": 12, "failed": 0, "errors": 0}


def test_tracer_smoke_writes_sanitized_evidence_with_fake_agent(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTRACE_NOTEBOOK_RUNTIME", "1")
    monkeypatch.setenv("AGENTRACE_RUN_RUNTIME", "1")
    monkeypatch.setenv("AGENTRACE_ALLOW_LIVE", "1")
    monkeypatch.setenv("AGENTRACE_EVIDENCE_DIR", str(tmp_path))

    class FakeApp:
        def invoke(self, state, config):
            tracer = config["callbacks"][0]
            tracer.trace_log.extend(
                [
                    {"type": "llm_start"},
                    {"type": "llm_end"},
                    {"type": "tool_start"},
                    {"type": "tool_end"},
                    {"type": "llm_end"},
                ]
            )

    monkeypatch.setattr(medical_agent_audit, "make_llm", lambda _: object())
    monkeypatch.setattr(medical_agent_audit, "build_tracer_smoke_agent", lambda _: FakeApp())
    medical_agent_audit.main()

    payload = json.loads(next(tmp_path.glob("*-tracer_smoke.json")).read_text(encoding="utf-8"))
    assert payload["status"] == "pass"
    assert payload["test_counts"] == {"events": 5, "llm_start_events": 1}
    assert "trace_log" not in payload


def test_tracer_smoke_preserves_sanitized_failed_event_counts(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTRACE_NOTEBOOK_RUNTIME", "1")
    monkeypatch.setenv("AGENTRACE_RUN_RUNTIME", "1")
    monkeypatch.setenv("AGENTRACE_ALLOW_LIVE", "1")
    monkeypatch.setenv("AGENTRACE_EVIDENCE_DIR", str(tmp_path))

    class ShortTraceApp:
        def invoke(self, state, config):
            config["callbacks"][0].trace_log.extend([{"type": "llm_start"}, {"type": "llm_end"}])

    monkeypatch.setattr(medical_agent_audit, "make_llm", lambda _: object())
    monkeypatch.setattr(medical_agent_audit, "build_tracer_smoke_agent", lambda _: ShortTraceApp())

    with pytest.raises(RuntimeError, match="Tracer smoke failed"):
        medical_agent_audit.main()

    payload = json.loads(next(tmp_path.glob("*-tracer_smoke.json")).read_text(encoding="utf-8"))
    assert payload["status"] == "fail"
    assert payload["test_counts"] == {"events": 2, "llm_start_events": 1}
    assert payload["failure_summary"] == "failure details withheld; inspect the notebook console"


def test_tracer_evidence_gate_rejects_malformed_evidence(tmp_path):
    (tmp_path / "runtime-20260821T000000Z-tracer_smoke.json").write_text("{}", encoding="utf-8")

    assert evidence.has_valid_tracer_evidence(tmp_path) is False


def test_stage_evidence_rejects_unsanitized_baseline_diagnostic_keys(tmp_path):
    with pytest.raises(ValueError, match="baseline diagnostic"):
        evidence.write_stage_evidence(
            stage="mini_gate",
            status="fail",
            test_counts={"records": 0, "condition_rows": 0},
            baseline_diagnostics={"model": {"raw_response": 1}},
            output_dir=tmp_path,
        )


def test_response_contract_probe_evidence_rejects_raw_payload_fields(tmp_path):
    with pytest.raises(ValueError, match="bounded result"):
        evidence.write_stage_evidence(
            stage="response_contract_probe",
            status="fail",
            test_counts={"models": 3, "passed": 0, "failed": 3},
            probe_diagnostics={
                "model": {"status": "fail", "category": "unknown", "calls": 1, "raw_response": "never-store"}
            },
            output_dir=tmp_path,
        )


def test_response_contract_probe_evidence_accepts_only_safe_request_failure_categories(tmp_path):
    path = evidence.write_stage_evidence(
        stage="response_contract_probe",
        status="fail",
        test_counts={"models": 3, "passed": 1, "failed": 2},
        probe_diagnostics={
            "model": {"status": "fail", "category": "response_format_rejected", "calls": 1}
        },
        output_dir=tmp_path,
    )

    assert "response_format_rejected" in path.read_text(encoding="utf-8")
