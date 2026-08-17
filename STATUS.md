# AgentTrace source status

**Updated:** 2026-08-18
**Current branch:** `main` (verify the exact HEAD with `git log -1`)
**Runtime status:** Static-only validation complete; notebook runtime has not been run.

## Done

- Local-first source scaffold and safety boundaries are present.
- Tracing, lexical detectors, sycophancy scoring, cache/study guards, NIST mapping, async API boundary, and guarded Gradio UI are implemented.
- Contract tests are written but intentionally unexecuted in the local workspace.
- README, contribution/security policy, GitHub templates, generated-results policy, docs index, methodology, and notebook instructions are present.
- Static verification passed: AST parsing for 24 Python files, expected-file inventory, credential scan, and `sqlite3` import scan.

## Next

Run `python notebooks/runtime_validation.py` only in Kaggle or Google Colab after setting `AGENTRACE_NOTEBOOK_RUNTIME=1`. This is the next task and is intentionally not run locally.

## Pending

- Notebook synthetic tests and tracer smoke test.
- Explicit live-API approval before the 45-call mini-gate.
- Full benchmark, W&B, deployment, publication, and empirical result artifacts.

## Evidence rule

This repository does not claim measured rates, confidence intervals, W&B links, live model availability, or deployment behavior until a dated notebook artifact verifies them.

For project rationale and authoritative contracts, read the sibling planning control plane at `../03-agentrace/STATUS.md`.
