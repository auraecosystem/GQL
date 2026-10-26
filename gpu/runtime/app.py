from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from runtime.hardware import detect_hardware
from runtime.jobs import create_job, get_job, list_jobs
from runtime.workers import WorkerUnavailable, worker_registry

app = FastAPI(
    title="GQL GPU Fabric",
    version="0.1.0",
    description=(
        "Multi-backend AI workload control-plane starter. Jobs are registered in memory; "
        "dispatch requires an explicitly registered trusted worker."
    ),
)


class JobRequest(BaseModel):
    kind: str = Field(description="inference, training, fine_tuning, evaluation, or embedding")
    model: str = Field(min_length=1, max_length=300)
    parameters: dict = Field(default_factory=dict)


@app.get("/health")
def health():
    return {"status": "ok", "service": "gql-gpu-fabric", "version": "0.1.0"}


@app.get("/hardware")
def hardware():
    return detect_hardware()


@app.get("/jobs")
def jobs():
    items = list_jobs()
    return {"items": items, "count": len(items)}


@app.post("/jobs", status_code=202)
def submit_job(request: JobRequest):
    try:
        return create_job(request.kind, request.model, request.parameters)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/jobs/{job_id}/dispatch")
def dispatch_job(job_id: str):
    """Dispatch through a trusted in-process adapter; no adapter is enabled by default."""
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    try:
        result = worker_registry.dispatch(job)
    except WorkerUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        # Do not leak adapter internals or stack traces to API clients.
        raise HTTPException(status_code=502, detail="Worker execution failed") from exc
    return {"job_id": job_id, "result": result}
