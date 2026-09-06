from agentrace.tracer import AgentTracer, _message_text


def test_smoke_agent_forces_two_safe_tool_calls_before_one_model_call():
    from langchain_core.messages import AIMessage

    from agentrace.agent import (
        TRACER_SMOKE_SYSTEM_PROMPT,
        TRACER_SMOKE_USER_PROMPT,
        build_tracer_smoke_agent,
    )

    class FakeLLM:
        def __init__(self):
            self.calls = []

        def invoke(self, messages):
            self.calls.append(messages)
            return AIMessage(content="A")

    llm = FakeLLM()
    result = build_tracer_smoke_agent(llm).invoke({"messages": [("user", "tracer smoke")]})
    tool_messages = [message for message in result["messages"] if message.type == "tool"]

    assert len(tool_messages) == 2
    assert all(message.content == "synthetic tracer smoke tool completed" for message in tool_messages)
    assert len(llm.calls) == 1
    assert "json" in TRACER_SMOKE_SYSTEM_PROMPT.lower()
    assert "json" in TRACER_SMOKE_USER_PROMPT.lower()
    assert llm.calls[0][0].content == TRACER_SMOKE_SYSTEM_PROMPT
    assert llm.calls[0][-1].content == TRACER_SMOKE_USER_PROMPT


def test_message_text_unwraps_chat_batches():
    assert _message_text([["system", "answer"], ["human", "A"]])


def test_tracer_normalizes_llm_start_handlers():
    tracer = AgentTracer()
    tracer.on_llm_start({}, ["prompt"], run_id="run-1")
    tracer.on_chat_model_start({}, [["prompt"]], run_id="run-2")
    assert [event["type"] for event in tracer.trace_log] == ["llm_start", "llm_start"]
