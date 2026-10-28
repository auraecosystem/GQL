"""Safe, pluggable worker dispatch contract.

Workers are trusted in-process adapters. Multiple adapters may serve the same job
kind; callers select one with parameters.worker when more than one is registered.
"""
from threading import RLock
from typing import Protocol, Any


class Worker(Protocol):
    """Trusted in-process adapter implemented by a separately reviewed backend."""

    def execute(self, job: dict[str, Any]) -> dict[str, Any]:
        """Execute one validated job and return a JSON-serializable result."""


class WorkerUnavailable(RuntimeError):
    """Raised when no suitable trusted worker is registered for a job."""


class WorkerRegistry:
    def __init__(self) -> None:
        self._workers: dict[str, dict[str, Worker]] = {}
        self._lock = RLock()

    def register(self, kind: str, worker: Worker, name: str | None = None) -> None:
        if not isinstance(kind, str) or not kind.strip():
            raise ValueError("Worker kind must be a non-empty string")
        if not callable(getattr(worker, "execute", None)):
            raise ValueError("Worker must implement execute(job)")
        worker_name = name or kind
        if not isinstance(worker_name, str) or not worker_name.strip():
            raise ValueError("Worker name must be a non-empty string")
        with self._lock:
            self._workers.setdefault(kind, {})[worker_name] = worker

    def unregister(self, kind: str, name: str | None = None) -> None:
        with self._lock:
            if name is None:
                self._workers.pop(kind, None)
                return
            named = self._workers.get(kind)
            if named is not None:
                named.pop(name, None)
                if not named:
                    self._workers.pop(kind, None)

    def available(self, kind: str) -> list[str]:
        with self._lock:
            return sorted(self._workers.get(kind, {}))

    def dispatch(self, job: dict[str, Any]) -> dict[str, Any]:
        kind = job.get("kind")
        parameters = job.get("parameters") or {}
        requested = parameters.get("worker") if isinstance(parameters, dict) else None
        with self._lock:
            named = dict(self._workers.get(kind, {}))
        if not named:
            raise WorkerUnavailable(f"No trusted worker is configured for job kind '{kind}'.")
        if requested is not None:
            if not isinstance(requested, str) or requested not in named:
                available = ", ".join(sorted(named))
                raise WorkerUnavailable(
                    f"Worker '{requested}' is not configured for '{kind}'. Available workers: {available}."
                )
            worker = named[requested]
        elif len(named) == 1:
            worker = next(iter(named.values()))
        else:
            available = ", ".join(sorted(named))
            raise WorkerUnavailable(
                f"Multiple workers support '{kind}'. Set parameters.worker to one of: {available}."
            )
        result = worker.execute(dict(job))
        if not isinstance(result, dict):
            raise TypeError("Worker result must be a dictionary")
        return result


# Deliberately empty until a trusted adapter is explicitly configured.
worker_registry = WorkerRegistry()
