"""Safe, pluggable worker dispatch contract.

No worker is enabled by default. This module deliberately does not execute arbitrary
commands, load user-supplied code, or pretend that GPU inference is available.
"""
from threading import RLock
from typing import Protocol, Any


class Worker(Protocol):
    """Trusted in-process adapter implemented by a separately reviewed backend."""

    def execute(self, job: dict[str, Any]) -> dict[str, Any]:
        """Execute one validated job and return a JSON-serializable result."""


class WorkerUnavailable(RuntimeError):
    """Raised when no trusted worker is registered for a job kind."""


class WorkerRegistry:
    def __init__(self) -> None:
        self._workers: dict[str, Worker] = {}
        self._lock = RLock()

    def register(self, kind: str, worker: Worker) -> None:
        if not kind or not isinstance(kind, str):
            raise ValueError("Worker kind must be a non-empty string")
        if not callable(getattr(worker, "execute", None)):
            raise ValueError("Worker must implement execute(job)")
        with self._lock:
            self._workers[kind] = worker

    def unregister(self, kind: str) -> None:
        with self._lock:
            self._workers.pop(kind, None)

    def dispatch(self, job: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            worker = self._workers.get(job.get("kind"))
        if worker is None:
            raise WorkerUnavailable(
                f"No trusted worker is configured for job kind '{job.get('kind')}'."
            )
        result = worker.execute(dict(job))
        if not isinstance(result, dict):
            raise TypeError("Worker result must be a dictionary")
        return result


# Deliberately empty until a trusted CPU/CUDA/ROCm adapter is configured.
worker_registry = WorkerRegistry()
