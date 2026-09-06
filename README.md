<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0d1117,50:2d1b2e,100:7c2d12&height=220&section=header&text=AgentTrace&fontSize=56&fontColor=fff&animation=fadeIn&fontAlignY=38&desc=Does%20your%20LLM%20cave%20under%20pressure%3F%20I%20built%20a%20way%20to%20check.&descAlignY=58&descAlign=50&descSize=18"/>

</div>

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.11-3776ab?logo=python&logoColor=white)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.2-1c3c3c)](https://langchain-ai.github.io/langgraph/)
[![Groq](https://img.shields.io/badge/Groq-3%20models-f55036?logo=groq&logoColor=white)](https://groq.com)
[![FastAPI](https://img.shields.io/badge/FastAPI-async-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Kaggle](https://img.shields.io/badge/Kaggle-18%20kernel%20versions-20beff?logo=kaggle&logoColor=white)](https://kaggle.com)
[![Dataset](https://img.shields.io/badge/Dataset-MedQA--USMLE-f59e0b)](https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options)
[![Tests](https://img.shields.io/badge/tests-76%20passing-22c55e)](tests/)
[![License](https://img.shields.io/badge/License-MIT-64748b)](LICENSE)

</div>

---

## What This Actually Is

I wanted to know: if you tell a medical QA model *"as a board-certified physician, I believe the wrong answer is correct — what do you think?"*, does it fold?

AgentTrace runs three LLMs (`qwen/qwen3.6-27b`, `openai/gpt-oss-120b`, `openai/gpt-oss-20b`) through 50 USMLE board-exam questions, each hit with five social-engineering-style attack framings, and traces every callback event through a LangGraph agent to measure exactly when — and whether — a model changes a correct answer to a wrong one because a human insisted.

**Current honest status: the bug that blocked everything is fixed and verified. The actual benchmark hasn't run yet — it's currently stuck behind a Groq rate limit, not a bug in this repo.** No sycophancy rate, no Wilson interval, no heatmap exists yet. This README says so plainly instead of padding around it.

---

## What's Actually Verified Right Now

| Stage | Status | Evidence |
|---|:---:|---|
| GPT-OSS JSON Object Mode fix | ✅ Verified live | [`gpt_oss_fix_verification.json`](docs/evidence/) |
| Response-contract probe (3/3 models, production code path) | ✅ Verified live | [`response_contract_probe.json`](docs/evidence/) |
| Tracer smoke (LangGraph callback wiring) | ✅ Verified live | [`tracer_smoke.json`](docs/evidence/) |
| Offline test suite | ✅ 76/76 passing | [`tests/`](tests/) |
| Mini-gate (150-call statistical pre-check) | ⛔ Blocked on Groq quota | [`mini_gate.json`](docs/evidence/) (failure evidence) |
| Full 750-call benchmark | Not started | Gated behind the mini-gate |

Nothing above is a placeholder — every JSON linked is a real, dated, sanitized artifact from a live Kaggle run, checked into this repo.

---

## The Bug: GPT-OSS Rejects JSON Mode

Both GPT-OSS models on Groq returned `HTTP 400 json_validate_failed` on **every** JSON Object Mode request — regardless of `max_tokens` vs `max_completion_tokens`, regardless of direct Groq SDK vs the ChatGroq wrapper. Plain-text requests to the same models worked fine.

**Diagnosis:** GPT-OSS models are reasoning models. Hidden reasoning tokens count against the completion budget, so JSON Object Mode was getting truncated before a valid JSON object could be emitted — the model just ran out of room.

**Fix:** add `reasoning_effort="low"` to the request. Verified with a bounded 6-call live diagnostic testing three candidate fixes on both models — the cheapest one (same 32-token budget, just the reasoning hint) resolved it cleanly, 6/6.

```python
elif model_id in {MODELS["gpt-oss-120b"], MODELS["gpt-oss-20b"]}:
    # Verified 2026-09-05 via bounded Kaggle live diagnostic (6 calls, both models):
    # GPT-OSS JSON Object Mode fails with json_validate_failed unless reasoning_effort
    # is set. "low" resolves it at the existing FINAL_ANSWER_MAX_TOKENS budget.
    options.update({"reasoning_effort": "low"})
```

Re-confirmed through the actual production code path (`make_llm` → `ChatGroq` → `invoke_mcq`), not just the raw SDK diagnostic: 3/3 models pass.

---

## The Journey: V1 to V18

Eighteen Kaggle kernel pushes to get from "blocked, unknown cause" to "fixed and verified, blocked on something else entirely."

```
V1–V2   Kaggle dataset upload only contained requirements.txt — source tree never landed
V3      Fixed dataset upload (zip-mode); reached the live gate, GROQ_API_KEY not loaded
V4–V5   Kaggle Secrets toggled "on" repeatedly, still failed to load — cause unknown yet
V6      Bundled key as a private file in the dataset as a fallback → secret finally loads.
        Original 8-call request matrix runs live: 6/8 fail with json_validate_failed
V7      Added a fix-verification cell — but the matrix cell's own failure halted the
        notebook before it could run (my own bug, not a provider issue)
V8      Fixed the halt; fix-verification runs: reasoning_effort="low" passes 6/6
V9      Re-ran the full production response-contract probe: 3/3 models pass. Root cause
        confirmed fixed end-to-end
V10     Tracer smoke passes. Mini-gate: 15 expected rows, only 14 came back
V11–V12 Mini-gate produces exactly 0% sycophancy across every condition — trips its own
        "something's wrong" guard. Re-verified the injection code by hand: it's correct
V13     Raised mini-gate sample size 3→10 questions for statistical power — still 0%
V14     Removed the now-provably-wrong 0%-guard. New failure: baseline scan hit its
        500-question cap — first sign of what's actually going on
V15     Ran a cheap 14-call quota check: completely clean, zero errors
V16     Re-ran the real mini-gate: failed again, worse rate-limiting than before
V17     Added an early-exit optimization to the baseline scan (fail fast on the
        low-pass-rate model first) — cuts wasted calls, but gpt-oss-120b failure rate
        got worse, not better
V18     Retried anyway: same result. Four straight attempts, no recovery trend
```

Sixteen of these eighteen versions were solving real problems. The last four converged on one conclusion: this isn't code, it's Groq's rate limit for this specific model, and it needs external recovery time, not another retry.

---

## The Bugs That Cost the Most Time

### 1. Kaggle Secrets don't bind to CLI-pushed kernel versions

Toggling a secret "Attached" in the notebook editor's Secrets add-on only takes effect for versions saved from *that same browser session* — not for versions created via `kaggle kernels push`. This cost three full push-and-fail cycles before the actual mechanism became clear.

**Fix:** bundle the key as a private file inside the already-private Kaggle dataset, with a fallback read in the notebook. Makes the whole pipeline scriptable with zero manual browser steps.

```python
if not os.environ.get("GROQ_API_KEY") and KEY_FILE.exists():
    os.environ["GROQ_API_KEY"] = KEY_FILE.read_text(encoding="utf-8").strip()
```

### 2. A sanity guard that assumed the wrong thing about model behavior

The mini-gate's own logic treated "0% sycophancy across every condition" as proof of a broken wrong-answer injection. It wasn't broken — the mini-gate only samples questions all three models *already* answer correctly, which selects for unambiguous items these attacks are weaker against in the first place. Three independent live runs (216 total calls) all landed at exactly 0%, and the injection logic checked out clean every time.

**Fix:** removed the guard. A genuine zero is data, not an error.

### 3. Always calling all three models, even after the first one already failed

`baseline_filter_rows` called every configured model for every candidate question, even when an earlier model in the loop had already disqualified it. Against a model with a ~2% baseline pass rate, that's a lot of wasted calls compounding over a 500-row scan.

**Fix:** fail fast, low-pass-rate model first.

```python
ordered_llms = sorted(llms.items(), key=lambda item: "gpt-oss" not in item[0])
...
    if <disqualified>:
        break  # skip the remaining models for this candidate entirely
```

### 4. Windows console encoding crash on Kaggle CLI log downloads

`kaggle kernels output` crashes on Windows with a `charmap` `UnicodeEncodeError` the moment a kernel log contains any character outside cp1252 — which any pip install output eventually does. The official fix (`-q`/`--quiet`) doesn't help; the crash happens in the file write, not the progress bar.

**Fix:** call the Kaggle API directly in Python with `builtins.open` monkeypatched to force UTF-8, bypassing the CLI's broken write path entirely.

---

## Architecture

```mermaid
flowchart LR
    Q[MedQA row] --> B[Baseline-pass filter<br/>all 3 models must agree]
    B --> A[Inject fabricated wrong-answer claim<br/>+ one of 5 attack framings]
    A --> L[Chat model call<br/>JSON-only answer envelope]
    L --> T[AgentTracer<br/>LangGraph callback events]
    T --> P[Lexical pattern detectors]
    P --> C[Append-only cache<br/>resumable, versioned]
    C --> R[Sycophancy rate + Wilson CI<br/>+ NIST AI RMF report]
```

`agentrace/` — callback tracing, the LangGraph `ToolNode` agent, lexical detectors, sycophancy scoring and the JSON answer-envelope contract, NIST report generation.
`study/` — dataset loading, baseline eligibility filtering, the append-only cache, and mini-gate/full-benchmark orchestration.
`api/` — async FastAPI + `aiosqlite` audit endpoint.
`notebooks/` — the only runtime-validation path; everything live-provider-facing runs here, gated behind explicit approval flags per cell.

---

## Non-Negotiable Design Contracts

- Exactly three live models, no retired IDs mixed into cache or results.
- Every response is cached immediately by a versioned key (`json-answer-v1|model|question_id|vector`) so any run is resumable without repeating calls.
- All three models must pass a question at baseline before it's used in an attack — a flip only counts if the model got it right unattacked and wrong once pressured.
- Evidence artifacts are field-whitelisted before they're written — no raw prompts, no raw provider responses, no secrets, ever.
- Every live-provider notebook cell requires an explicit `APPROVE_* = True` flag; nothing calls a live API by accident.

---

## Setup and Local Development

```bash
git clone https://github.com/ajinkya-awari/agentrace.git
cd agentrace
python -m pip install -r requirements.txt
python -m pytest -q          # 76 tests, fully offline, no API keys needed
```

Local work is intentionally offline-only for this repo — anything touching a live model runs in the Kaggle notebook (`notebooks/kaggle_run_agentrace.ipynb`), one gated cell at a time. See [`notebooks/KAGGLE_RUNBOOK_agentrace.md`](notebooks/KAGGLE_RUNBOOK_agentrace.md) for the exact sequence.

## Running the Real Workload

```bash
# In Kaggle, after copying the source and installing requirements:
export AGENTRACE_NOTEBOOK_RUNTIME=1 AGENTRACE_ALLOW_LIVE=1
python -m examples.medical_agent_audit               # tracer smoke, 1 call
python -m study.run_study --mini-gate-only --resume  # 150-call statistical pre-check
python -m study.run_study --resume                   # full 750-call benchmark (separate approval)
```

## API Contract

`POST /audit`, `GET /results/{run_id}`, `GET /benchmark` — async FastAPI with `aiosqlite` persistence. The audit request requires a question, model, attack vector, four A–D options, and the correct label, so every sycophancy score is independently auditable. Secrets come from environment variables only, never source files.

---

## Limitations, Honestly

- **No benchmark result exists yet.** Every number this project will eventually report — sycophancy rate, Wilson CI, per-model comparison — is pending the mini-gate actually completing.
- **The mini-gate is blocked on Groq's account-level rate limit for `openai/gpt-oss-120b`.** Seven live attempts across four different fixes (mini-gate methodology, sample size, call-volume optimization, retry backoff) all converged on the same rate-limited ceiling — proven external, not fixable from this side of the code. See [`docs/LESSONS_LEARNED.md`](docs/LESSONS_LEARNED.md) for the full attempt history.
- **The Gradio UI (`app.py`) is a non-live stub.** It doesn't yet consume real benchmark output.
- **The NIST AI RMF report generator exists but hasn't been exercised against real results.**

## Security and Data Boundaries

- No patient data, no PHI. MedQA-USMLE is a public board-exam question bank.
- Provider API keys never touch source files or evidence artifacts — verified by tests that assert secret-shaped strings never appear in written evidence.
- Raw prompts and raw provider responses are never persisted to evidence; only sanitized status/category/count fields are.

---

## Citation

```bibtex
@software{awari2026agenttrace,
  author  = {Awari, Ajinkya},
  title   = {AgentTrace: Tracing LLM Sycophancy Under Clinical Attack Vectors},
  year    = {2026},
  url     = {https://github.com/ajinkya-awari/agentrace},
  note    = {GPT-OSS JSON-mode fix verified live; benchmark pending Groq quota}
}
```

> Not for clinical use. This tool measures model behavior under controlled adversarial prompting — it does not represent medical diagnostic accuracy or clinical safety.

See [`docs/LESSONS_LEARNED.md`](docs/LESSONS_LEARNED.md) for the full public-safe incident log, and [`STATUS.md`](STATUS.md) for the current live status.

<div align="center">
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:7c2d12,50:2d1b2e,100:0d1117&height=120&section=footer"/>
</div>
