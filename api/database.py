"""Async SQLite persistence; synchronous sqlite3 is intentionally not used."""

import json
from typing import Any

try:
    import aiosqlite
except ImportError:  # Local static contracts must not require dependency installation.
    class _MissingAioSqlite:
        Row = dict

        @staticmethod
        def connect(*args, **kwargs):
            raise RuntimeError("aiosqlite is required for database operations in the notebook runtime")

    aiosqlite = _MissingAioSqlite()  # type: ignore[assignment]

DB_PATH = "agentrace.db"


async def init_db() -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS audit_runs (
                run_id TEXT PRIMARY KEY,
                model TEXT NOT NULL,
                question TEXT NOT NULL,
                vector TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                trace_json TEXT NOT NULL,
                syco_score INTEGER NOT NULL,
                nist_json TEXT NOT NULL
            )
            """
        )
        await db.commit()


async def insert_audit_run(record: dict[str, Any]) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """
            INSERT INTO audit_runs
            (run_id, model, question, vector, timestamp, trace_json, syco_score, nist_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record["run_id"], record["model"], record["question"], record["vector"],
                record["timestamp"], json.dumps(record["trace"]), int(record["sycophancy_detected"]),
                json.dumps(record["nist_report"]),
            ),
        )
        await db.commit()


async def get_audit_run(run_id: str) -> dict[str, Any] | None:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT * FROM audit_runs WHERE run_id = ?", (run_id,)) as cursor:
            row = await cursor.fetchone()
    return None if row is None else dict(row)
