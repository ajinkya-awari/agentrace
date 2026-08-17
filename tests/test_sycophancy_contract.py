from agentrace.sycophancy import (
    ATTACK_TEMPLATES,
    get_wrong_option,
    parse_answer,
    run_attack,
    wilson_ci,
)


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
    assert 5.0 < lo < 10.0 < hi < 20.0
