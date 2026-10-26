from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from runtime.hardware import detect_hardware
from runtime.jobs import create_job, list_jobs

app = FastAPI(
    title="GQL GPU Fabric",
    version="0.1.0",
    description="Multi-backend AI workload control-plane starter. Job registration does not execute workloads.",
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
