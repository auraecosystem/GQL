"""In-memory workload request registry; does not execute submitted work."""
from datetime import datetime, timezone
from threading import Lock
from uuid import uuid4

ALLOWED_KINDS = {"inference", "training", "fine_tuning", "evaluation", "embedding"}
_jobs = []
_lock = Lock()


def create_job(kind, model, parameters=None):
    if kind not in ALLOWED_KINDS:
        raise ValueError(f"Unsupported job kind: {kind}")
    job = {
        "id": str(uuid4()),
        "kind": kind,
        "model": model,
        "parameters": dict(parameters or {}),
        "status": "accepted",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "note": "Registered only; no worker is configured to execute this job.",
    }
    with _lock:
        _jobs.append(job)
    return dict(job)


def list_jobs():
    with _lock:
        return [dict(job, parameters=dict(job["parameters"])) for job in _jobs]


def clear_jobs_for_tests():
    with _lock:
        _jobs.clear()
