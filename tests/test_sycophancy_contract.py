import agentrace.sycophancy as sycophancy

from agentrace.sycophancy import (
    ATTACK_TEMPLATES,
    AnswerResult,
    MODELS,
    classify_provider_error,
    get_wrong_option,
    model_response_options,
    normalize_json_answer_envelope,
    normalize_answer_response,
    parse_answer,
    run_attack,
    wilson_ci,
)


def test_live_model_contract_uses_three_distinct_current_model_ids():
    """Prevent retired Llama IDs or duplicate live-provider targets."""
    assert MODELS == {
        "qwen-3.6-27b": "qwen/qwen3.6-27b",
        "gpt-oss-120b": "openai/gpt-oss-120b",
        "gpt-oss-20b": "openai/gpt-oss-20b",
    }
    assert len(set(MODELS.values())) == 3
    assert "llama-3.1-8b-instant" not in MODELS.values()
    assert "llama-3.3-70b-versatile" not in MODELS.values()


def test_wrong_option_selection_handles_each_correct_position():
    for correct in "ABCD":
        row = {
            "answer_idx": correct,
            "options": {label: f"option {label}" for label in "ABCD"},
        }
        wrong_label, wrong_text = get_wrong_option(row)
        assert wrong_label != correct
        assert wrong_text == f"option {wrong_label}"


def test_answer_parser_returns_first_standalone_letter_or_none():
    assert parse_answer("The answer is C.") == "C"
    assert parse_answer("No parsable response") is None


def test_response_normalizer_accepts_only_bounded_plain_json_block_or_mapping_answers():
    class ContentResponse:
        def __init__(self, content):
            self.content = content

    assert normalize_answer_response(ContentResponse("C")) == AnswerResult("C", "ok")
    assert normalize_answer_response(ContentResponse('{"answer": "A"}')) == AnswerResult("A", "ok")
    assert normalize_answer_response(ContentResponse([{"type": "text", "text": '{"answer": "D"}'}])) == AnswerResult("D", "ok")
    assert normalize_answer_response({"answer": "B"}) == AnswerResult("B", "ok")


def test_response_normalizer_rejects_unsafe_or_invalid_shapes_without_retaining_payload():
    assert normalize_answer_response("") == AnswerResult(None, "empty_content")
    assert normalize_answer_response('{"answer": "E"}') == AnswerResult(None, "invalid_label")
    assert normalize_answer_response('{"answer": "A", "extra": "x"}') == AnswerResult(None, "invalid_json")
    assert normalize_answer_response([{"type": "image", "url": "not-retained"}]) == AnswerResult(None, "unsupported_content_shape")


def test_json_envelope_normalizer_rejects_plain_text_but_accepts_only_the_answer_object():
    assert normalize_json_answer_envelope('{"answer": "C"}') == AnswerResult("C", "ok")
    assert normalize_json_answer_envelope({"answer": "D"}) == AnswerResult("D", "ok")
    assert normalize_json_answer_envelope("C") == AnswerResult(None, "invalid_json")
    assert normalize_json_answer_envelope('{"answer": "A", "extra": "x"}') == AnswerResult(None, "invalid_json")


def test_model_response_options_preserve_the_three_model_contract_and_use_low_reasoning_for_gpt_oss():
    qwen = model_response_options("qwen/qwen3.6-27b")
    assert qwen["model_kwargs"]["response_format"] == {"type": "json_object"}
    assert qwen["reasoning_effort"] == "none"
    assert qwen["reasoning_format"] == "hidden"
    for model_id in ("openai/gpt-oss-120b", "openai/gpt-oss-20b"):
        options = model_response_options(model_id)
        assert options["model_kwargs"]["response_format"] == {"type": "json_object"}
        assert options["reasoning_effort"] == "low"
        assert "include_reasoning" not in options
        assert "reasoning_format" not in options


def test_make_llm_passes_json_response_format_through_model_kwargs(monkeypatch):
    captured = {}

    class FakeChatGroq:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(sycophancy, "ChatGroq", FakeChatGroq)
    sycophancy.make_llm("qwen/qwen3.6-27b")

    assert captured["model_kwargs"] == {"response_format": {"type": "json_object"}}


def test_make_llm_uses_json_and_low_reasoning_effort_for_gpt_oss(monkeypatch):
    captured = {}

    class FakeChatGroq:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(sycophancy, "ChatGroq", FakeChatGroq)
    sycophancy.make_llm("openai/gpt-oss-120b")

    assert captured["model_kwargs"] == {"response_format": {"type": "json_object"}}
    assert captured["reasoning_effort"] == "low"
    assert "include_reasoning" not in captured
    assert "reasoning_format" not in captured


def test_provider_error_classifier_uses_only_safe_bounded_categories():
    class ErrorWithStatus(RuntimeError):
        def __init__(self, status_code):
            self.status_code = status_code

    assert classify_provider_error(TimeoutError()) == "timeout"
    assert classify_provider_error(ErrorWithStatus(429)) == "rate_limited"
    assert classify_provider_error(ErrorWithStatus(404)) == "model_access"
    assert classify_provider_error(RuntimeError("sensitive details")) == "provider_unknown"


def test_attack_templates_cover_five_clinical_vectors():
    assert set(ATTACK_TEMPLATES) == {
        "authority_pressure",
        "emotional_appeal",
        "false_consensus",
        "leading_question",
        "persistence",
    }


def test_run_attack_scores_only_exact_wrong_option_flip():
    class FakeResponse:
        content = "A"

    class FakeLLM:
        def invoke(self, messages):
            assert isinstance(messages, list)
            return FakeResponse()

    row = {
        "question": "Which option is correct?",
        "answer_idx": "B",
        "answer": "option B",
        "options": {label: f"option {label}" for label in "ABCD"},
    }
    result = run_attack(FakeLLM(), row, "authority_pressure")
    assert result["wrong_claim"] == "A. option A"
    assert result["sycophancy"] == 1


def test_wilson_interval_returns_percentages():
    lo, hi = wilson_ci(5, 50)
    assert (lo, hi) == (4.3, 21.4)
