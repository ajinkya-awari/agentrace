# AgentTrace source status

**Audit date:** 2026-09-07
**Status label:** **GPT-OSS fix verified and live on GitHub; mini-gate blocked on external Groq rate limit**
**Overall completion:** **68%** portfolio readiness
**Confidence:** High for the fix and all code changes (live-verified); the mini-gate blocker is
proven external (Groq rate/quota ceiling for `openai/gpt-oss-120b`), not a code defect.

**Public repository:** https://github.com/ajinkya-awari/agentrace (renamed from the earlier
`-agentrace`). README rewritten with the real V1-V21 Kaggle debugging journey, honest current
status, LICENSE added, sanitized dated evidence checked into `docs/evidence/`.

This is the runtime source repository for Project 03. The planning/control authority with full
attempt-by-attempt detail is the sibling folder:

`E:\application\MS CS\portfolio-projects\03-agentrace` (see `STATUS.md`, `HANDOVER.md`,
`NEXT_SESSION_HANDOFF.md` there for the complete seven-attempt mini-gate history).

## What changed since the 2026-08-29 audit

- GPT-OSS JSON Object Mode bug isolated, fixed (`reasoning_effort="low"`), and verified through
  the real production `make_llm` path (3/3 models pass).
- Tracer smoke passes live.
- Mini-gate methodology corrected (false zero-rate guard removed, sample size raised, baseline-scan
  call volume optimized) — but the mini-gate itself has not passed. Seven live attempts all hit the
  same Groq-side rate limit for `openai/gpt-oss-120b`; a `max_retries` increase (4→10) proved this
  is not client-side-fixable.
- Full offline test suite: 76/76 passing.
- Pushed to GitHub with a rewritten, honest README.

## Current implementation state

Implemented source exists for:

- Callback tracing and LangGraph ToolNode smoke path.
- Lexical pattern detectors.
- Sycophancy scoring helpers, JSON answer envelope normalization, provider error categorization, and Wilson intervals.
- Current provider model map:
  - `qwen/qwen3.6-27b`
  - `openai/gpt-oss-120b`
  - `openai/gpt-oss-20b`
- Dataset normalization and strict all-three-correct baseline eligibility.
- Append-only cache with `json-answer-v1` cache version.
- Mini-gate orchestration, `--mini-gate-only`, tracer-evidence gate, and cache reuse.
- Sanitized runtime evidence writer.
- Response-contract probe and bounded GPT-OSS request-shape matrix.
- FastAPI/aiosqlite boundary, Gradio app scaffold, GitHub docs/templates, Kaggle notebook, and runbook.

## Evidence and execution state

- Local static source evidence: `docs/evidence/2026-08-21-source-static.txt`.
- Local handoff-hardening evidence: `docs/evidence/2026-08-21-handoff-hardening-static.txt`.
- Documented Kaggle synthetic evidence: `docs/evidence/runtime-20260822T162352Z-synthetic_validation.json`, 66 tests passed.
- Documented response-contract probe: `docs/evidence/runtime-20260822T162507Z-response_contract_probe.json`, Qwen passed and both GPT-OSS models failed with sanitized `invalid_request`.

The 2026-08-22 runtime JSON files are not present in this local checkout inspected on 2026-08-29. Treat them as historical Kaggle evidence references unless the artifacts are present in the active Kaggle working directory.

## Not verified

- GPT-OSS request compatibility.
- Current all-model response-contract probe pass.
- Current tracer smoke.
- Mini-gate pass.
- Full benchmark.
- Wilson interval table, heatmap, NIST report, W&B, Hugging Face, deployment, publication, email, or GitHub release.

## Next task

In Kaggle, after uploading the latest source, run only:

```bash
PYTHONPATH=/kaggle/working/agentrace python /kaggle/working/agentrace/notebooks/gpt_oss_request_matrix.py
```

Stop after the matrix and inspect the dated sanitized `runtime-*-gpt_oss_request_matrix.json` artifact. Do not run tracer smoke, mini-gate, or the full benchmark until the matrix proves a targeted fix and the fix is reviewed.

## Local laptop policy

Do not run Python, pytest, pip, Kaggle commands, provider/API calls, data/model downloads, GPU/hard CPU work, tracer smoke, mini-gate, benchmark, W&B, Hugging Face, deployment, publication, email, commit, or push locally. Local work is limited to reading, documentation/source edits, static scans, notebook JSON validation, Git status/diff inspection, and other non-runtime checks.
