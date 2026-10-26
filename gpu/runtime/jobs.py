"""In-memory workload request registry; does not execute submitted work."""
from copy import deepcopy
from datetime import datetime, timezone
from threading import Lock
from uuid import uuid4

ALLOWED_KINDS = {"inference", "training", "fine_tuning", "evaluation", "embedding"}
MAX_JOBS = 1000
_jobs = []
_lock = Lock()


def create_job(kind, model, parameters=None):
    if kind not in ALLOWED_KINDS:
        raise ValueError(f"Unsupported job kind: {kind}")
    job = {
        "id": str(uuid4()),
        "kind": kind,
        "model": model,
        "parameters": deepcopy(parameters or {}),
        "status": "accepted",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "note": "Registered only; no worker is configured to execute this job.",
    }
    with _lock:
        _jobs.append(job)
        if len(_jobs) > MAX_JOBS:
            del _jobs[: len(_jobs) - MAX_JOBS]
    return deepcopy(job)


def list_jobs():
    with _lock:
        return deepcopy(_jobs)


def clear_jobs_for_tests():
    with _lock:
        _jobs.clear()
