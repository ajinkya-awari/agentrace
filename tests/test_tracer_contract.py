from agentrace.tracer import AgentTracer, _message_text


def test_message_text_unwraps_chat_batches():
    assert _message_text([["system", "answer"], ["human", "A"]])


def test_tracer_normalizes_llm_start_handlers():
    tracer = AgentTracer()
    tracer.on_llm_start({}, ["prompt"], run_id="run-1")
    tracer.on_chat_model_start({}, [["prompt"]], run_id="run-2")
    assert [event["type"] for event in tracer.trace_log] == ["llm_start", "llm_start"]
