"""Notebook-friendly validation gate; never run this from the local workspace."""

import os
import subprocess
import sys


def run() -> None:
    if not os.getenv("AGENTRACE_NOTEBOOK_RUNTIME"):
        raise RuntimeError("Run runtime_validation.py only inside Kaggle or Google Colab")
    subprocess.run([sys.executable, "-m", "pytest", "-q"], check=True)
    if os.getenv("AGENTRACE_ALLOW_LIVE") != "1":
        print("Synthetic checks complete; live provider gates remain closed.")
        return
    os.environ.setdefault("AGENTRACE_RUN_RUNTIME", "1")
    subprocess.run([sys.executable, "-m", "examples.medical_agent_audit"], check=True)


if __name__ == "__main__":
    run()
