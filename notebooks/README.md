# Notebook execution

This is the next task recorded in [STATUS.md](../STATUS.md). Do not run this path in the local workspace.

Runtime validation and any approved benchmark execution will be performed in Kaggle or Google Colab, not on the local CPU workspace.

## Current blocked preflight

The normal notebook sequence is paused. The only approved live action is the bounded GPT-OSS
request-shape matrix:

`PYTHONPATH=/kaggle/working/agentrace python notebooks/gpt_oss_request_matrix.py`

Run it only after the reviewed source is uploaded to Kaggle, copied into `/kaggle/working/agentrace`,
and live approval flags plus the Kaggle-held `GROQ_API_KEY` are set. It makes at most eight hosted
calls and writes sanitized request-matrix evidence. Do not run tracer smoke, the mini-gate, or the
full benchmark until this evidence identifies a proven GPT-OSS request fix.

## Normal sequence after a proven fix

1. **Dependency install**: in `/kaggle/working/agentrace`, run `python -m pip install -r requirements.txt`.
2. **Synthetic validation**: set `AGENTRACE_NOTEBOOK_RUNTIME=1` with all live-provider flags unset, then run `PYTHONPATH=/kaggle/working/agentrace python notebooks/runtime_validation.py`.
3. **Response-contract probe**: only after synthetic validation passes, retrieve `GROQ_API_KEY` from Kaggle Secrets, set `AGENTRACE_ALLOW_LIVE=1`, then run `PYTHONPATH=/kaggle/working/agentrace python notebooks/response_contract_probe.py`.
4. **Tracer smoke**: only after probe evidence passes, run `PYTHONPATH=/kaggle/working/agentrace python -m examples.medical_agent_audit`.
5. **Mini-gate**: only after passing tracer evidence and separate approval, run `PYTHONPATH=/kaggle/working/agentrace python -m study.run_study --mini-gate-only --resume`.
6. **Full benchmark**: run only after explicit approval following mini-gate evidence review.

`runtime_validation.py` is provider-free and writes a sanitized dated evidence record. The
response-contract probe calls each configured model exactly once using a non-clinical synthetic MCQ;
it validates the JSON envelope and writes only model identifiers, bounded status/category/counts,
and the active response/cache/request-profile contract. Qwen uses its documented hidden-reasoning
JSON profile; GPT-OSS uses JSON Object Mode only, without adapter-forwarded reasoning fields. It
exits non-zero if any model fails. Neither stage starts
tracer smoke, the mini-gate, or the full benchmark.

## Current preflight block (2026-08-22)

Synthetic validation passed 66 tests at `docs/evidence/runtime-20260822T162352Z-synthetic_validation.json`.
The response-contract probe then passed Qwen but recorded `invalid_request` for both GPT-OSS slots,
one call each, at `docs/evidence/runtime-20260822T162507Z-response_contract_probe.json`. Do not
run tracer smoke. The next approved action is only
`PYTHONPATH=/kaggle/working/agentrace python notebooks/gpt_oss_request_matrix.py`, after a new
dataset version is attached and live approval is explicitly set. It makes at most eight calls and
writes sanitized request-matrix evidence; it does not alter production configuration.

## Current live-model contract

The three distinct benchmark IDs are `qwen/qwen3.6-27b`, `openai/gpt-oss-120b`, and
`openai/gpt-oss-20b`. `qwen/qwen3.6-27b` is a preview model, so the first post-update tracer smoke
uses Qwen and must prove account access. The previous failed tracer artifact for the retired
`llama-3.1-8b-instant` ID remains preserved and is not valid gate evidence. Do not combine results
or cache entries from retired models with this contract.

The GPU is not used to accelerate Groq API calls; it provides an isolated execution environment and is not required for these stages. Notebook files must not contain API keys, raw provider responses, or unpublished benchmark findings.
