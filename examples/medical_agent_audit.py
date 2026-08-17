"""Trace one medical agent run when explicitly invoked in the notebook runtime."""

import os

from agentrace.agent import build_medical_agent
from agentrace.sycophancy import MODELS, make_llm
from agentrace.tracer import AgentTracer


def main() -> None:
    if not os.getenv("AGENTRACE_RUN_RUNTIME"):
        raise RuntimeError("Runtime examples are notebook-gated; set AGENTRACE_RUN_RUNTIME=1 there")
    llm = make_llm(MODELS["llama-3.1-8b"])
    app = build_medical_agent(llm)
    tracer = AgentTracer()
    app.invoke(
        {"messages": [("user", "Which option is correct: A, B, C, or D?")]},
        config={"callbacks": [tracer]},
    )
    for event in tracer.trace_log:
        print(event["type"], event["run_id"], event["ts"])
    print(f"Total events: {len(tracer.trace_log)}")
    assert any(event["type"] == "llm_start" for event in tracer.trace_log)


if __name__ == "__main__":
    main()
