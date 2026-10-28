"""SQLite-backed workload registry with atomic lifecycle transitions."""
import json
import os
import sqlite3
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from uuid import uuid4

ALLOWED_KINDS = {"inference", "training", "fine_tuning", "evaluation", "embedding", "script_analysis"}
MAX_JOBS = 1000
_lock = RLock()


def _database_path():
    # Containers set GQL_JOB_DB to the persistent /cache volume.
    return os.environ.get("GQL_JOB_DB", "/tmp/gql-gpu-fabric-jobs.sqlite3")


def _connect():
    path = _database_path()
    if path != ":memory:":
        Path(path).expanduser().parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute(
        """CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            status TEXT NOT NULL,
            payload TEXT NOT NULL
        )"""
    )
    return connection


@contextmanager
def _database():
    connection = _connect()
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def _decode(row):
    if row is None:
        return None
    job = json.loads(row["payload"])
    job["status"] = row["status"]
    return job


def create_job(kind, model, parameters=None):
    if kind not in ALLOWED_KINDS:
        raise ValueError(f"Unsupported job kind: {kind}")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("Model must be a non-empty string")
    if parameters is None:
        parameters = {}
    if not isinstance(parameters, dict):
        raise ValueError("Parameters must be an object")

    job = {
        "id": str(uuid4()),
        "kind": kind,
        "model": model.strip(),
        "parameters": deepcopy(parameters),
        "status": "accepted",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "note": "Registered only; dispatch requires a trusted worker adapter.",
    }
    payload = json.dumps(job, separators=(",", ":"), ensure_ascii=False)
    with _lock, _database() as connection:
        connection.execute(
            "INSERT INTO jobs (id, created_at, status, payload) VALUES (?, ?, ?, ?)",
            (job["id"], job["created_at"], job["status"], payload),
        )
        connection.execute(
            "DELETE FROM jobs WHERE id IN (SELECT id FROM jobs ORDER BY created_at DESC LIMIT -1 OFFSET ?)",
            (MAX_JOBS,),
        )
    return deepcopy(job)


def get_job(job_id):
    with _lock, _database() as connection:
        row = connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return deepcopy(_decode(row))


def list_jobs():
    with _lock, _database() as connection:
        rows = connection.execute("SELECT * FROM jobs ORDER BY created_at ASC").fetchall()
        return deepcopy([_decode(row) for row in rows])


def claim_job(job_id):
    """Atomically claim an accepted job so concurrent requests cannot run it twice."""
    with _lock, _database() as connection:
        cursor = connection.execute(
            "UPDATE jobs SET status = 'running' WHERE id = ? AND status = 'accepted'",
            (job_id,),
        )
        return cursor.rowcount == 1


def update_job(job_id, expected_status, status, **fields):
    """Transition a job only if it is still in the expected lifecycle state."""
    with _lock, _database() as connection:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None or row["status"] != expected_status:
            return False
        job = json.loads(row["payload"])
        job.update(fields)
        job["status"] = status
        cursor = connection.execute(
            "UPDATE jobs SET status = ?, payload = ? WHERE id = ? AND status = ?",
            (
                status,
                json.dumps(job, separators=(",", ":"), ensure_ascii=False),
                job_id,
                expected_status,
            ),
        )
        return cursor.rowcount == 1


def clear_jobs_for_tests():
    with _lock, _database() as connection:
        connection.execute("DELETE FROM jobs")
