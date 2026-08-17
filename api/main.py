"""FastAPI routes with explicit provider and database boundaries."""

import json
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, HTTPException

from agentrace.nist_report import NISTReportGenerator
from agentrace.sycophancy import MODELS, OPTION_LABELS, make_llm, run_attack
from api.database import get_audit_run, init_db, insert_audit_run
from api.models import AuditRequest, AuditResponse


@asynccontextmanager
async def lifespan(application):
    await init_db()
    yield


app = FastAPI(title="AgentTrace Audit API", lifespan=lifespan)


@app.post("/audit", response_model=AuditResponse)
async def audit(request: AuditRequest):
    if request.model not in MODELS:
        raise HTTPException(status_code=400, detail="Unknown model")
    if not os.getenv("GROQ_API_KEY"):
        raise HTTPException(status_code=503, detail="GROQ_API_KEY is not configured")
    if request.correct_label not in OPTION_LABELS or request.options is None or set(request.options) != set(OPTION_LABELS):
        raise HTTPException(status_code=422, detail="A-D options and correct_label are required for an auditable MCQ")
    llm = make_llm(MODELS[request.model])
    row = {
        "question": request.question,
        "answer_idx": request.correct_label,
        "answer": request.options[request.correct_label],
        "options": request.options,
    }
    result = run_attack(llm, row, request.attack_vector)
    report = NISTReportGenerator().generate(float(result["sycophancy"] * 100), {})
    record = {
        "run_id": str(uuid4()),
        "model": request.model,
        "question": request.question,
        "vector": request.attack_vector,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "trace": [],
        "sycophancy_detected": bool(result["sycophancy"]),
        "nist_report": report,
    }
    await insert_audit_run(record)
    return {"run_id": record["run_id"], "sycophancy_detected": record["sycophancy_detected"], "nist_report": report}


@app.get("/results/{run_id}")
async def results(run_id: str):
    record = await get_audit_run(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return record


@app.get("/benchmark")
async def benchmark():
    path = Path("study/results/sycophancy_table.json")
    if not path.exists():
        raise HTTPException(status_code=404, detail="Benchmark results are not available")
    return json.loads(path.read_text(encoding="utf-8"))
