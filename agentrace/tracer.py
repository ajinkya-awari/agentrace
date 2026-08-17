"""Callback normalization for append-only AgentTrace events."""

from datetime import datetime, timezone
from typing import Any

try:
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.messages import BaseMessage
except ImportError:  # Allows static-only local inspection without dependencies.
    class BaseCallbackHandler:  # type: ignore[no-redef]
        pass

    class BaseMessage:  # type: ignore[no-redef]
        content: Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _message_text(messages: Any) -> str:
    """Normalize chat callback batches and message-like values to text."""
    if messages and isinstance(messages[0], list):
        messages = messages[0]
    parts = []
    for message in messages or []:
        if isinstance(message, BaseMessage):
            parts.append(str(message.content))
        elif isinstance(message, tuple) and len(message) == 2:
            parts.append(str(message[1]))
        else:
            parts.append(str(message))
    return "\n".join(parts)


class AgentTracer(BaseCallbackHandler):
    """Collect normalized callback events without mutating provider state."""

    def __init__(self) -> None:
        self.trace_log: list[dict[str, Any]] = []

    def on_llm_start(self, serialized: dict, prompts: list[str], **kwargs: Any) -> None:
        self.trace_log.append(
            {
                "type": "llm_start",
                "prompts": [str(prompt) for prompt in prompts],
                "run_id": str(kwargs.get("run_id")),
                "ts": _now(),
            }
        )

    def on_chat_model_start(self, serialized: dict, messages: Any, **kwargs: Any) -> None:
        self.trace_log.append(
            {
                "type": "llm_start",
                "prompts": [_message_text(messages)],
                "run_id": str(kwargs.get("run_id")),
                "ts": _now(),
            }
        )

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        text = ""
        generations = getattr(response, "generations", None) or []
        if generations and generations[0]:
            generation = generations[0][0]
            text = getattr(generation, "text", "")
            if not text and getattr(generation, "message", None) is not None:
                text = getattr(generation.message, "content", "")
        self.trace_log.append(
            {
                "type": "llm_end",
                "output": str(text),
                "run_id": str(kwargs.get("run_id")),
                "ts": _now(),
            }
        )

    def on_tool_start(self, serialized: dict, input_str: str, **kwargs: Any) -> None:
        self.trace_log.append(
            {
                "type": "tool_start",
                "tool": serialized.get("name", "unknown_tool"),
                "input": kwargs.get("inputs", input_str),
                "run_id": str(kwargs.get("run_id")),
                "ts": _now(),
            }
        )

    def on_tool_end(self, output: Any, **kwargs: Any) -> None:
        value = getattr(output, "content", output)
        self.trace_log.append(
            {
                "type": "tool_end",
                "output": str(value),
                "run_id": str(kwargs.get("run_id")),
                "ts": _now(),
            }
        )
