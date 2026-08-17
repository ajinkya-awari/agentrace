# Contributing to AgentTrace

AgentTrace is developed local-first with runtime validation isolated to Kaggle or Google Colab.

## Before opening a change

- Read `notebooks/README.md` and the project architecture in `README.md`.
- Keep provider calls, dataset downloads, W&B uploads, deployment, and generated evidence behind their explicit gates.
- Never add API keys, raw provider responses, restricted data, or fabricated empirical findings.
- Keep changes focused and document contract changes in the planning repository.

## Validation policy

The local workspace is limited to source inspection, documentation, AST/static checks, and diff review. Do not run tests, install dependencies, download data, call models, or run CPU-heavy workloads locally. Run the contract suite only from the guarded notebook runtime.

## Pull requests

Describe the changed contract, list the static checks performed, and state clearly whether notebook runtime validation was performed. A runtime result must include its dated artifact path and gate status.
