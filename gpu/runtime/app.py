from pathlib import Path
from typing import Any
import os
import secrets

from fastapi import Body, FastAPI, Header, HTTPException
from graphql import GraphQLError, build_schema, graphql_sync
from graphql.language import value_from_ast_untyped
from pydantic import BaseModel, Field

from runtime.hardware import detect_hardware
from runtime.diagnostics import collect_diagnostics
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


def _configure_optional_workers() -> None:
    """Register local inference only after an explicit operator opt-in."""
    if os.getenv("GQL_ENABLE_LOCAL_TRANSFORMERS_WORKER", "").strip().lower() in {"1", "true", "yes"}:
        from runtime.local_worker import LocalTransformersWorker
        worker_registry.register("inference", LocalTransformersWorker())


_configure_optional_workers()

app = FastAPI(
    title="GQL GPU Fabric",
    version="0.3.0",
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


def _json_scalar_value(value: Any) -> Any:
    """Accept JSON-compatible values only; reject Python objects from resolvers."""
    import json

    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise GraphQLError("Value is not JSON serializable") from exc


graphql_schema_path = Path(__file__).resolve().parents[1] / "schema" / "schema.gql"
graphql_schema = build_schema(graphql_schema_path.read_text(encoding="utf-8"))
json_scalar = graphql_schema.type_map["JSON"]
json_scalar.serialize = _json_scalar_value
json_scalar.parse_value = _json_scalar_value
json_scalar.parse_literal = lambda node, variables=None: _json_scalar_value(
    value_from_ast_untyped(node, variables)
)


def _graphql_health(_source, _info):
    return {"status": "ok", "service": "gql-gpu-fabric", "version": app.version}


def _graphql_hardware(_source, _info):
    detected = detect_hardware()
    torch_info = detected.get("torch") or {}
    return {
        "platform": detected.get("platform", "unknown"),
        "python": detected.get("python", "unknown"),
        "backend_hint": detected.get("backend_hint", "cpu_or_gpu_not_exposed"),
        "rocminfo_detected": bool(detected.get("rocminfo_detected", False)),
        "nvidia_smi": detected.get("nvidia_smi"),
        "model_dir": detected.get("model_dir", "/models"),
        "torch": {
            "available": bool(torch_info.get("available", False)),
            "cuda_available": bool(torch_info.get("cuda_available", False)),
            "device_count": int(torch_info.get("device_count", 0)),
            "version": torch_info.get("version"),
        },
    }


def _graphql_register_job(_source, _info, input):
    try:
        return create_job(input["kind"], input["model"], input.get("parameters") or {})
    except ValueError as exc:
        raise GraphQLError(str(exc)) from exc


graphql_schema.type_map["Query"].fields["health"].resolve = _graphql_health
graphql_schema.type_map["Query"].fields["hardware"].resolve = _graphql_hardware
graphql_schema.type_map["Query"].fields["jobs"].resolve = lambda _source, _info: list_jobs()
graphql_schema.type_map["Mutation"].fields["registerJob"].resolve = _graphql_register_job


@app.get("/health")
def health():
    return {"status": "ok", "service": "gql-gpu-fabric", "version": app.version}


@app.get("/hardware")
def hardware():
    return detect_hardware()


@app.get("/diagnostics")
def diagnostics():
    """Return read-only runtime, GPU visibility, and mount diagnostics."""
    return collect_diagnostics()


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


@app.post("/graphql")
def graphql_endpoint(payload: dict = Body(...)):
    """Execute a GraphQL query or mutation over a JSON POST request."""
    query = payload.get("query")
    if not isinstance(query, str) or not query.strip():
        raise HTTPException(status_code=400, detail="'query' must be a non-empty string")
    if len(query) > 100_000:
        raise HTTPException(status_code=413, detail="GraphQL query is too large")

    variables = payload.get("variables")
    if variables is not None and not isinstance(variables, dict):
        raise HTTPException(status_code=400, detail="'variables' must be a JSON object")
    operation_name = payload.get("operationName")
    if operation_name is not None and not isinstance(operation_name, str):
        raise HTTPException(status_code=400, detail="'operationName' must be a string")

    result = graphql_sync(
        graphql_schema,
        query,
        variable_values=variables,
        operation_name=operation_name,
    )
    response = {"data": result.data}
    if result.errors:
        response["errors"] = [
            {
                "message": error.message,
                **({"path": error.path} if error.path else {}),
            }
            for error in result.errors
        ]
    return response


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
    """Authenticated synchronous inference endpoint over a trusted worker.

    This endpoint is disabled until an operator configures GQL_API_TOKEN.
    Existing job-control and GraphQL routes remain separate.
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
