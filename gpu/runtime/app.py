import os
import secrets

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from runtime.hardware import detect_hardware
from runtime.jobs import (
    claim_job,
    create_job,
    get_job,
    list_jobs,
    update_job,
)
from runtime.workers import WorkerUnavailable, worker_registry
from runtime.yolov7 import YOLOv7ONNXWorker
from runtime.caffe_ssd import CaffeSSDWorker

app = FastAPI(
    title="GQL GPU Fabric",
    version="0.2.0",
    description=(
        "Multi-backend AI workload control plane with persistent job state. "
        "Execution requires an explicitly registered trusted worker."
    ),
)

# Explicit opt-in prevents enabling inference against an unreviewed model directory.
if os.getenv("GQL_ENABLE_YOLOV7_WORKER", "").lower() in {"1", "true", "yes"}:
    worker_registry.register("inference", YOLOv7ONNXWorker())

# Use a separate opt-in for Caffe SSD; it takes precedence if both are enabled.
if os.getenv("GQL_ENABLE_CAFFE_SSD_WORKER", "").lower() in {"1", "true", "yes"}:
    worker_registry.register("inference", CaffeSSDWorker())


class JobRequest(BaseModel):
    kind: str = Field(description="inference, training, fine_tuning, evaluation, or embedding")
    model: str = Field(min_length=1, max_length=300)
    parameters: dict = Field(default_factory=dict)


@app.get("/health")
def health():
    return {"status": "ok", "service": "gql-gpu-fabric", "version": "0.2.0"}


@app.get("/hardware")
def hardware():
    return detect_hardware()


@app.get("/jobs")
def jobs():
    items = list_jobs()
    return {"items": items, "count": len(items)}


@app.get("/jobs/{job_id}")
def job_detail(job_id: str):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.post("/jobs", status_code=202)
def submit_job(request: JobRequest):
    try:
        return create_job(request.kind, request.model, request.parameters)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/jobs/{job_id}/dispatch")
def dispatch_job(job_id: str):
    """Run a job once through a trusted adapter and persist its lifecycle state."""
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    if not claim_job(job_id):
        raise HTTPException(
            status_code=409,
            detail=f"Job cannot be dispatched from status '{get_job(job_id)['status']}'.",
        )

    job = get_job(job_id)
    try:
        result = worker_registry.dispatch(job)
    except WorkerUnavailable as exc:
        update_job(
            job_id,
            expected_status="running",
            status="accepted",
            note=str(exc),
        )
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        update_job(
            job_id,
            expected_status="running",
            status="failed",
            error="Worker execution failed",
        )
        # Keep worker internals and stack traces out of the API response.
        raise HTTPException(status_code=502, detail="Worker execution failed") from exc

    update_job(
        job_id,
        expected_status="running",
        status="completed",
        result=result,
    )
    return {"job_id": job_id, "status": "completed", "result": result}


@app.post("/v1/inference")
def serve_inference(
    request: JobRequest,
    authorization: str | None = Header(default=None),
):
    """Authenticated synchronous inference endpoint over a registered trusted worker.

    Set GQL_API_TOKEN to enable this endpoint. The existing job-control endpoints
    remain separate and should be protected at the deployment boundary.
    """
    configured_token = os.getenv("GQL_API_TOKEN", "")
    if not configured_token:
        raise HTTPException(status_code=503, detail="Inference API is disabled; configure GQL_API_TOKEN")
    expected = f"Bearer {configured_token}"
    if not authorization or not secrets.compare_digest(authorization, expected):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if request.kind != "inference":
        raise HTTPException(status_code=422, detail="This endpoint accepts inference jobs only")
    try:
        job = create_job(request.kind, request.model, request.parameters)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return dispatch_job(job["id"])
