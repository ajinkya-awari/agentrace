from agentrace.patterns import (
    PatternDetector,
    bow_cosine,
    detect_circular_reasoning,
    detect_ignored_output,
    detect_repeated_tool_calls,
)


def test_bow_cosine_is_lexical_and_empty_safe():
    assert bow_cosine("same clinical question", "same clinical question") == 1.0
    assert bow_cosine("", "same clinical question") == 0.0


def test_circular_reasoning_requires_three_mutually_similar_prompts():
    prompts = ["repeat the clinical question now"] * 3
    assert detect_circular_reasoning(prompts) == [(0, 1, 2)]


def test_repeated_tool_calls_flag_three_matching_inputs():
    trace = [
        {"type": "tool_start", "tool": "lookup", "input": {"q": "x"}},
        {"type": "tool_start", "tool": "lookup", "input": {"q": "x"}},
        {"type": "tool_start", "tool": "lookup", "input": {"q": "x"}},
    ]
    flagged = detect_repeated_tool_calls(trace)
    assert flagged[0]["count"] == 3


def test_ignored_output_flags_output_missing_from_next_prompt():
    trace = [
        {"type": "tool_end", "output": "reference result"},
        {"type": "llm_start", "prompts": ["new unrelated question"]},
    ]
    assert detect_ignored_output(trace)[0]["severity"] == "MEDIUM"


def test_pattern_detector_returns_all_empty_lists_for_clean_trace():
    assert PatternDetector().analyze([]) == {
        "repeated_tool_calls": [],
        "ignored_outputs": [],
        "circular_reasoning": [],
    }
