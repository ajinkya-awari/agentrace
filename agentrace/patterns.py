"""Pure-Python lexical and trace misuse detectors."""

from collections import Counter, defaultdict
from itertools import combinations
import math
import re
from typing import Any

TOKEN_RE = re.compile(r"[a-z0-9]+")


def bow(text: str) -> Counter[str]:
    return Counter(TOKEN_RE.findall(str(text).lower()))


def bow_cosine(a: str, b: str) -> float:
    left, right = bow(a), bow(b)
    common = set(left) & set(right)
    dot = sum(left[token] * right[token] for token in common)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    return 0.0 if left_norm == 0 or right_norm == 0 else dot / (left_norm * right_norm)


def detect_circular_reasoning(
    prompts: list[str], threshold: float = 0.75
) -> list[tuple[int, int, int]]:
    flagged = []
    for i, j, k in combinations(range(len(prompts)), 3):
        similarities = (
            bow_cosine(prompts[i], prompts[j]),
            bow_cosine(prompts[i], prompts[k]),
            bow_cosine(prompts[j], prompts[k]),
        )
        if min(similarities) >= threshold:
            flagged.append((i, j, k))
    return flagged


def _stable_value(value: Any) -> str:
    return repr(value)


def detect_repeated_tool_calls(trace_log: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, event in enumerate(trace_log):
        if event.get("type") == "tool_start":
            key = (str(event.get("tool", "unknown_tool")), _stable_value(event.get("input", "")))
            grouped[key].append(index)
    return [
        {"tool": tool, "input": input_value, "count": len(indices), "indices": indices, "severity": "HIGH"}
        for (tool, input_value), indices in grouped.items()
        if len(indices) >= 3
    ]


def detect_ignored_output(trace_log: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flagged = []
    for index, event in enumerate(trace_log[:-1]):
        if event.get("type") != "tool_end":
            continue
        following = trace_log[index + 1]
        if following.get("type") != "llm_start":
            continue
        output = str(event.get("output", ""))
        prompts = "\n".join(str(prompt) for prompt in following.get("prompts", []))
        if output and output not in prompts:
            flagged.append({"index": index, "output": output, "severity": "MEDIUM"})
    return flagged


class PatternDetector:
    def __init__(self, threshold: float = 0.75) -> None:
        self.threshold = threshold

    def analyze(self, trace_log: list[dict[str, Any]]) -> dict[str, list[Any]]:
        prompts = [
            str(prompt)
            for event in trace_log
            if event.get("type") == "llm_start"
            for prompt in event.get("prompts", [])
        ]
        return {
            "repeated_tool_calls": detect_repeated_tool_calls(trace_log),
            "ignored_outputs": detect_ignored_output(trace_log),
            "circular_reasoning": detect_circular_reasoning(prompts, self.threshold),
        }
