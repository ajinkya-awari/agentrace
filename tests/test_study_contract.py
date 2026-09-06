import json

import pytest

from study.cache import cache_key, load_cache, save_result
from study import run_study
from study.run_study import MINI_GATE_QUESTION_COUNT, dry_run_plan, run_mini_gate
from study.dataset import BaselineEligibilityError, baseline_filter_rows
from agentrace.sycophancy import AnswerResult


def test_cache_round_trip_uses_stable_key(tmp_path):
    path = tmp_path / "raw.jsonl"
    record = {"model": "model-a", "question_id": "q1", "vector": "persistence", "sycophancy": 0}
    save_result(record, path)
    assert cache_key("model-a", "q1", "persistence") in load_cache(path)


def test_dry_run_has_no_live_api_and_counts_mini_gate():
    plan = dry_run_plan(max_questions=50)
    assert plan["live_api"] is False
    assert plan["mini_gate_calls"] == MINI_GATE_QUESTION_COUNT * 3 * 5
    assert plan["full_gate_calls"] == 750


def test_cached_record_is_json_serializable(tmp_path):
    path = tmp_path / "raw.jsonl"
    save_result({"model": "m", "question_id": "q", "vector": "v", "value": 1}, path)
    json.dumps(next(iter(load_cache(path).values())))


def test_cache_keys_version_the_response_contract_without_deleting_prior_records():
    assert cache_key("model-a", "q1", "persistence").startswith("json-answer-v1|")


def test_mini_gate_requires_all_fifteen_condition_rows():
    records = [
        {
            "model": model,
            "vector": vector,
            "attacked_answer": "A",
            "sycophancy": index % 2,
        }
        for model in ("m1", "m2", "m3")
        for vector in ("v1", "v2", "v3", "v4", "v5")
        for index in range(MINI_GATE_QUESTION_COUNT)
    ]
    assert len(run_mini_gate(records)) == 15


def _mini_records():
    return [
        {"model": model, "vector": vector, "attacked_answer": "A", "sycophancy": index % 2}
        for model in run_study.MODELS
        for vector in run_study.ATTACK_TEMPLATES
        for index in range(MINI_GATE_QUESTION_COUNT)
    ]


def test_mini_gate_only_never_invokes_the_full_benchmark():
    calls = []

    def fake_benchmark(questions, resume):
        calls.append((list(questions), resume))
        return _mini_records()

    rows = run_study.run_live_study(
        questions=[{"question_id": str(index)} for index in range(MINI_GATE_QUESTION_COUNT)],
        resume=True,
        mini_gate_only=True,
        benchmark_runner=fake_benchmark,
    )

    assert len(rows) == 15
    assert calls == [([{"question_id": str(index)} for index in range(MINI_GATE_QUESTION_COUNT)], True)]


def test_full_run_reuses_the_mini_gate_cache():
    calls = []

    def fake_benchmark(questions, resume):
        calls.append((list(questions), resume))
        return _mini_records()

    run_study.run_live_study(
        questions=[{"question_id": str(index)} for index in range(5)],
        resume=False,
        mini_gate_only=False,
        benchmark_runner=fake_benchmark,
        output_writer=lambda records, table: None,
        wandb_logger=lambda table: None,
    )

    assert calls[0][1] is False
    assert calls[1][1] is True


def test_live_mini_gate_refuses_without_valid_tracer_evidence(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTRACE_NOTEBOOK_RUNTIME", "1")
    monkeypatch.setenv("AGENTRACE_ALLOW_LIVE", "1")
    monkeypatch.setattr(run_study, "EVIDENCE_DIR", tmp_path)

    with pytest.raises(RuntimeError, match="tracer smoke evidence"):
        run_study.main(["--mini-gate-only"])


def test_main_requests_configured_baselines_for_mini_gate_and_preserves_resume(monkeypatch):
    monkeypatch.setenv("AGENTRACE_NOTEBOOK_RUNTIME", "1")
    monkeypatch.setenv("AGENTRACE_ALLOW_LIVE", "1")
    monkeypatch.setattr(run_study, "has_valid_tracer_evidence", lambda _: True)
    monkeypatch.setattr(run_study, "make_llm", lambda model_id: object())
    requested = []
    received = []

    def fake_load_baseline_questions(llms, n):
        requested.append(n)
        return [{"question_id": str(index)} for index in range(n)]

    monkeypatch.setattr("study.dataset.load_baseline_questions", fake_load_baseline_questions)
    monkeypatch.setattr(
        run_study,
        "run_live_study",
        lambda *, questions, resume, mini_gate_only: received.append((len(questions), resume, mini_gate_only)) or [],
    )
    monkeypatch.setattr(run_study, "write_stage_evidence", lambda **_: "evidence.json")

    run_study.main(["--mini-gate-only", "--resume"])

    assert requested == [MINI_GATE_QUESTION_COUNT]
    assert received == [(MINI_GATE_QUESTION_COUNT, True, True)]


def test_main_requests_configured_baselines_for_full_mode(monkeypatch):
    monkeypatch.setenv("AGENTRACE_NOTEBOOK_RUNTIME", "1")
    monkeypatch.setenv("AGENTRACE_ALLOW_LIVE", "1")
    monkeypatch.setattr(run_study, "has_valid_tracer_evidence", lambda _: True)
    monkeypatch.setattr(run_study, "make_llm", lambda model_id: object())
    requested = []

    def fake_load_baseline_questions(llms, n):
        requested.append(n)
        return [{"question_id": str(index)} for index in range(n)]

    monkeypatch.setattr("study.dataset.load_baseline_questions", fake_load_baseline_questions)
    monkeypatch.setattr(run_study, "run_live_study", lambda **_: [])

    run_study.main(["--max-questions", "7"])

    assert requested == [7]


def test_baseline_filter_counts_a_correct_answer_and_marks_row_eligible():
    rows = [{"id": "q1", "question": "q", "answer_idx": "A", "options": {key: key for key in "ABCD"}}]
    passing, counts = baseline_filter_rows(rows, {"correct": "correct"}, 1, invoke=lambda llm, *_: "A")
    assert passing == [{"id": "q1", "question": "q", "answer_idx": "A", "options": {key: key for key in "ABCD"}, "question_id": "q1"}]
    assert counts["correct"]["correct_answers"] == 1


def test_baseline_filter_counts_an_incorrect_answer_and_stops_checking_remaining_models():
    rows = [{"id": "q1", "question": "q", "answer_idx": "A", "options": {key: key for key in "ABCD"}}]
    answers = {"incorrect": "B", "never_called": "A"}
    passing, counts = baseline_filter_rows(rows, {name: name for name in answers}, 1, invoke=lambda llm, *_: answers[llm])
    assert passing == []
    assert counts["incorrect"]["incorrect_answers"] == 1
    assert counts["never_called"]["total_calls"] == 0


def test_baseline_filter_counts_an_unparseable_answer():
    rows = [{"id": "q1", "question": "q", "answer_idx": "A", "options": {key: key for key in "ABCD"}}]
    passing, counts = baseline_filter_rows(rows, {"unparseable": "unparseable"}, 1, invoke=lambda llm, *_: None)
    assert passing == []
    assert counts["unparseable"]["unparseable_answers"] == 1


def test_baseline_filter_counts_a_provider_error_from_a_raised_exception():
    rows = [{"id": "q1", "question": "q", "answer_idx": "A", "options": {key: key for key in "ABCD"}}]

    def fake_invoke(llm, question, options):
        raise RuntimeError("provider")

    passing, counts = baseline_filter_rows(rows, {"error": "error"}, 1, invoke=fake_invoke)
    assert passing == []
    assert counts["error"]["provider_errors"] == 1


def test_baseline_filter_preserves_strict_eligibility_and_safe_error_categories():
    rows = [{"id": "q1", "question": "q", "answer_idx": "A", "options": {key: key for key in "ABCD"}}]
    answers = {
        "parse": AnswerResult(None, "invalid_json"),
        "never_called": AnswerResult("A", "ok"),
    }

    passing, counts = baseline_filter_rows(rows, {name: name for name in answers}, 1, invoke=lambda llm, *_: answers[llm])

    assert passing == []
    assert counts["parse"]["unparseable_answers"] == 1
    assert counts["parse"]["parse_invalid_json"] == 1
    assert counts["never_called"]["total_calls"] == 0


def test_baseline_filter_checks_gpt_oss_models_before_qwen_to_fail_fast():
    rows = [{"id": "q1", "question": "q", "answer_idx": "A", "options": {key: key for key in "ABCD"}}]
    answers = {"qwen-3.6-27b": AnswerResult("A", "ok"), "gpt-oss-120b": AnswerResult(None, "rate_limited")}

    passing, counts = baseline_filter_rows(rows, {name: name for name in answers}, 1, invoke=lambda llm, *_: answers[llm])

    assert passing == []
    assert counts["gpt-oss-120b"]["total_calls"] == 1
    assert counts["qwen-3.6-27b"]["total_calls"] == 0


def test_zero_eligible_baseline_failure_evidence_includes_sanitized_counts(monkeypatch):
    error = BaselineEligibilityError({"m": {"total_calls": 1, "parsable_answers": 0, "correct_answers": 0, "incorrect_answers": 0, "unparseable_answers": 1, "provider_errors": 0}})
    monkeypatch.setenv("AGENTRACE_NOTEBOOK_RUNTIME", "1")
    monkeypatch.setenv("AGENTRACE_ALLOW_LIVE", "1")
    monkeypatch.setattr(run_study, "has_valid_tracer_evidence", lambda _: True)
    monkeypatch.setattr(run_study, "make_llm", lambda _: object())
    monkeypatch.setattr("study.dataset.load_baseline_questions", lambda *_, **__: (_ for _ in ()).throw(error))
    captured = {}
    monkeypatch.setattr(run_study, "write_stage_evidence", lambda **kwargs: captured.update(kwargs) or "evidence.json")

    with pytest.raises(RuntimeError, match="Mini-gate failed"):
        run_study.main(["--mini-gate-only"])

    assert captured["baseline_diagnostics"] == error.diagnostics
