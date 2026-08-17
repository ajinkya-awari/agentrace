# Notebook execution

Runtime validation and any approved benchmark execution will be performed in Kaggle or Google Colab, not on the local CPU workspace.

1. Create a fresh Kaggle/Colab notebook with GPU runtime enabled.
2. Clone this repository into notebook-local storage.
3. Install `requirements.txt` only inside the notebook environment.
4. Set `AGENTRACE_NOTEBOOK_RUNTIME=1` for synthetic/runtime checks.
5. Set `AGENTRACE_ALLOW_LIVE=1` only after explicit approval for provider calls.
6. Use `runtime_validation.py` for the guarded sequence.

The GPU is not used to accelerate Groq API calls; it provides an isolated execution environment. Notebook files must not contain API keys, raw provider responses, or unpublished benchmark findings.
