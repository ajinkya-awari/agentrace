"""Append-only JSONL cache for resumable provider results."""

import json
from pathlib import Path
from typing import Any

CACHE_PATH = Path("study/results/raw_responses.jsonl")
RESPONSE_CONTRACT_VERSION = "json-answer-v1"


def cache_key(model: str, q_id: str, vector: str) -> str:
    return f"{RESPONSE_CONTRACT_VERSION}|{model}|{q_id}|{vector}"


def load_cache(path: Path = CACHE_PATH) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    records: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                if "key" in record:
                    records[record["key"]] = record
    return records


def save_result(record: dict[str, Any], path: Path = CACHE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    value = dict(record)
    value.setdefault("key", cache_key(value["model"], value["question_id"], value["vector"]))
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")
