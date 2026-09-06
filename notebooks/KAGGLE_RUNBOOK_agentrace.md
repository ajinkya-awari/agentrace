# AgentTrace Kaggle Runbook

This runbook is for the upload-ready Kaggle notebook:

`notebooks/kaggle_run_agentrace.ipynb`

Do not run runtime commands on the Windows laptop. All dependency installation, Python validation,
provider calls, dataset access, and benchmark work belong in Kaggle.

## Current State

- Project slug: `agentrace`
- Kaggle dataset title: `AgentTrace Source Private`
- Expected Kaggle dataset slug: `agenttrace-source-private`
- Kaggle source path: detected by the notebook from `/kaggle/input`; do not hard-code it.
- Working copy path: `/kaggle/working/agentrace`
- Evidence directory: `/kaggle/working/agentrace/docs/evidence`
- Synthetic validation evidence reference: `docs/evidence/runtime-20260822T162352Z-synthetic_validation.json`
- Response-contract probe evidence reference: `docs/evidence/runtime-20260822T162507Z-response_contract_probe.json`

Verified state from Project 03 records:

- Synthetic validation passed with 66 tests.
- Qwen response-contract probe passed.
- GPT-OSS 20B and GPT-OSS 120B response probes failed with sanitized `invalid_request`.
- A prior tracer smoke passed, but the current JSON response-contract preflight blocks rerunning it.
- Mini-gate and full benchmark have not passed and must not be claimed.

## Attach the Kaggle Dataset

1. Open the Kaggle notebook.
2. In the right sidebar, open **Input**.
3. Click **Add Input**.
4. Search for the private dataset titled `AgentTrace Source Private`.
5. Attach the newest dataset version.
6. Do not attach datasets containing raw responses, generated benchmark results, credentials, or
   unrelated projects.

The notebook detects the actual source tree by checking for:

- `requirements.txt`
- `agentrace/`
- `api/`
- `examples/`
- `notebooks/`
- `study/`
- `tests/`

## Cell-by-Cell Use

Run one cell at a time. Read the printed status and next action before running the next cell.

1. Cell 1 is safe. It prints project configuration and current blockers. It does not access secrets.
2. Cell 2 is safe. It inspects `/kaggle/input` and detects the attached source tree.
3. Cell 3 is safe. It copies the complete source tree to `/kaggle/working/agentrace`. If that
   directory already exists, it moves it to a timestamped backup instead of deleting it.
4. Cell 4 is safe. It validates required files in the working copy.
5. Cell 5 installs dependencies in Kaggle only. Record dependency conflict warnings; do not hide
   them or change requirements solely because Kaggle has preinstalled package conflicts.
6. Cell 6 tells you whether to restart the kernel. It does not restart automatically.
7. Cell 7 is safe. It sets provider-free synthetic-mode environment variables and removes live flags.
8. Cell 8 runs provider-free synthetic validation only. It writes dated sanitized evidence and stops
   on failure.
9. Cell 9 reads only the latest sanitized synthetic evidence. It prints status, counts, timestamp,
   and failure summary category only.
10. Cell 10 is a live-access stop gate. It does not load secrets.
11. Cell 11 loads `GROQ_API_KEY` from Kaggle Secrets and prints only `secret loaded: True/False`.
12. Cell 12 is the first cell that requires explicit approval. Edit
   `APPROVE_LIVE_PREFLIGHT = False` to `True` before running it. It runs only the bounded GPT-OSS
   request-shape matrix and stops afterward.
13. Cell 13 prints the next gated command. It does not execute it unless you edit the guard.

## Safe Cells

Cells 1 through 10 are safe from provider access. Cell 5 installs dependencies and must run only in
Kaggle. Cells 7 through 9 are provider-free.

## Approval Cells

- Cell 11 requires a Kaggle Secret named `GROQ_API_KEY`; it does not print the value.
- Cell 12 requires explicit approval and runs at most the documented GPT-OSS request matrix.
- Cell 13 is a final command gate and must remain disabled until prior evidence is reviewed.

## Commands That Must Never Run Automatically

- Tracer smoke: `PYTHONPATH=/kaggle/working/agentrace python -m examples.medical_agent_audit`
- Mini-gate: `PYTHONPATH=/kaggle/working/agentrace python -m study.run_study --mini-gate-only --resume`
- Full benchmark: any full `study.run_study` command without `--mini-gate-only`
- W&B upload, Hugging Face deployment, email, Git push, or publication

## Evidence Preservation

Evidence is saved under:

`/kaggle/working/agentrace/docs/evidence`

The notebook never deletes evidence or working directories. It moves any existing working copy to:

`/kaggle/working/agentrace_backup_<timestamp>`

After a failed run:

1. Do not rerun repeatedly.
2. Use the evidence inspection cell for sanitized metadata only.
3. Download or save the dated `runtime-*.json` file if you need to preserve it outside the notebook.
4. Do not expose raw provider responses, prompts, API keys, patient data, or exception text.

## Current Blocker

Project 03 is blocked at the GPT-OSS response-contract preflight. The next approved action is only:

`PYTHONPATH=/kaggle/working/agentrace python notebooks/gpt_oss_request_matrix.py`

The matrix is an eight-call maximum diagnostic for the two GPT-OSS models and four request shapes.
It exists because Qwen already passed the response-contract probe. It must produce dated sanitized
evidence before any targeted provider request-profile fix is considered.

## After the Matrix

If the matrix fails, stop. Report only sanitized model, profile, status, category, call count, and
evidence path.

If the matrix passes, stop. Do not run tracer smoke or mini-gate automatically. A targeted source
review is still required before continuing to synthetic validation, response-contract probe, tracer
smoke, and mini-gate.
