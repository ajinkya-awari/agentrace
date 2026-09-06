"""Notebook-friendly validation gate; never run this from the local workspace."""

import os
import subprocess
import sys

from agentrace.evidence import write_stage_evidence


def _test_counts(output: str) -> dict[str, int]:
    import re

    return {
        "passed": int((match.group(1) if (match := re.search(r"(\d+) passed", output)) else 0)),
        "failed": int((match.group(1) if (match := re.search(r"(\d+) failed", output)) else 0)),
        "errors": int((match.group(1) if (match := re.search(r"(\d+) error", output)) else 0)),
    }


def run(command_runner=subprocess.run) -> None:
    if os.getenv("AGENTRACE_NOTEBOOK_RUNTIME") != "1":
        raise RuntimeError("Run runtime_validation.py only inside Kaggle or Google Colab")
    result = command_runner(
        [sys.executable, "-m", "pytest", "-q"],
        check=False,
        capture_output=True,
        text=True,
    )
    output = f"{result.stdout}\n{result.stderr}"
    print(output, end="")
    counts = _test_counts(output)
    if result.returncode == 0:
        path = write_stage_evidence(stage="synthetic_validation", status="pass", test_counts=counts)
        print(f"Synthetic checks complete; live provider gates remain closed. Evidence: {path}")
        return
    path = write_stage_evidence(
        stage="synthetic_validation",
        status="fail",
        test_counts=counts,
        failure_summary=f"pytest return code {result.returncode}",
    )
    raise RuntimeError(f"Synthetic validation failed. Evidence: {path}")


if __name__ == "__main__":
    run()
