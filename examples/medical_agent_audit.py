"""Trace one medical agent run when explicitly invoked in the notebook runtime."""

import os

from agentrace.agent import build_tracer_smoke_agent
from agentrace.evidence import write_stage_evidence
from agentrace.sycophancy import MODELS, make_llm
from agentrace.tracer import AgentTracer


def main() -> None:
    if os.getenv("AGENTRACE_NOTEBOOK_RUNTIME") != "1" or os.getenv("AGENTRACE_RUN_RUNTIME") != "1":
        raise RuntimeError("Tracer smoke is notebook-gated; set AGENTRACE_NOTEBOOK_RUNTIME=1 and AGENTRACE_RUN_RUNTIME=1")
    if os.getenv("AGENTRACE_ALLOW_LIVE") != "1":
        raise RuntimeError("Tracer smoke requires explicit live approval via AGENTRACE_ALLOW_LIVE=1")
    event_count = 0
    llm_start_count = 0
    try:
        # Qwen is a preview-model access gate; the smoke test must prove this account can use it.
        llm = make_llm(MODELS["qwen-3.6-27b"])
        app = build_tracer_smoke_agent(llm)
        tracer = AgentTracer()
        app.invoke(
            {"messages": []},
            config={"callbacks": [tracer]},
        )
        event_count = len(tracer.trace_log)
        llm_start_count = sum(event.get("type") == "llm_start" for event in tracer.trace_log)
        assert event_count >= 5
        assert llm_start_count >= 1
        path = write_stage_evidence(
            stage="tracer_smoke",
            status="pass",
            test_counts={"events": event_count, "llm_start_events": llm_start_count},
        )
        print(f"Tracer smoke passed. Evidence: {path}")
    except Exception as exc:
        path = write_stage_evidence(
            stage="tracer_smoke",
            status="fail",
            test_counts={"events": event_count, "llm_start_events": llm_start_count},
            failure_summary=type(exc).__name__,
        )
        raise RuntimeError(f"Tracer smoke failed. Evidence: {path}") from exc


if __name__ == "__main__":
    main()
