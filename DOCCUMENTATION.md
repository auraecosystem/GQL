```py
import os, textwrap, zipfile, subprocess, sys, json, pathlib

root = pathlib.Path("/mnt/data/gql_gpu_fabric")
files = {
"README.md": r"""
# GQL GPU Fabric

A starter multi-backend runtime scaffold for AI model workloads: NVIDIA CUDA, AMD ROCm, CPU fallback, training jobs, inference integration points, GraphQL schema, Conda environment, and datacenter deployment.

> This is an executable starter scaffold, not a complete model-serving or distributed training system. Model loading/training adapters are intentionally explicit extension points.

## Layout
- `gpu/Dockerfile.gql` — default NVIDIA CUDA-based application image
- `gpu/Dockerfile.cuda` — NVIDIA variant
- `gpu/Dockerfile.rocm` — AMD ROCm variant
- `gpu/Dockerfile.cpu` — CPU fallback
- `gpu/runtime/` — FastAPI service, hardware detection, job registry
- `gpu/schema/schema.gql` — GraphQL schema contract
- `gpu/config/runtime.gpu` — runtime profile in YAML syntax
- `gpu/environment.conda.yml` — Conda environment definition
- `datacenter/compose.yaml` — local service deployment example
- `tests/` — standard-library unit tests
```
## Quick start (CPU)
```bash
cd gpu
python -m venv .venv
. .venv/bin/activate        # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
uvicorn runtime.app:app --host 127.0.0.1 --port 8080
```
Then open /health, /hardware, or /docs.

Build the NVIDIA image

From repository root:
```bash
docker build -f gpu/Dockerfile.gql -t gql-gpu-fabric:cuda .
docker run --rm -p 8080:8080 --gpus all gql-gpu-fabric:cuda
```
The Docker build requires internet access for the base image and Python packages. The host needs a compatible NVIDIA driver and NVIDIA Container Toolkit. Choose a CUDA/framework combination compatible with your model and GPU.

Other backends
AMD: build gpu/Dockerfile.rocm on a supported Linux host with a compatible ROCm installation/host setup.
CPU: docker build -f gpu/Dockerfile.cpu -t gql-gpu-fabric:cpu .
Conda: conda env create -f gpu/environment.conda.yml

The CUDA, ROCm, and CPU images are separate on purpose; one container cannot safely assume their runtimes are interchangeable.
```bash
Validation
python -m unittest discover -s tests -v
python -m compileall -q gpu/runtime tests
API
GET /health — liveness
GET /hardware — detected hardware capabilities
GET /jobs — in-memory job records
POST /jobs — register a workload request (does not launch expensive training)
GET /docs — interactive API documentation
```
Security notes: this scaffold binds to all interfaces inside the container. Put it behind an authenticated gateway before exposing it to a network. Job records are in-memory and reset on restart. Do not mount untrusted model files or expose the Docker socket to the container.
""",
```"gpu/Dockerfile.gql": r"""

GQL default image: NVIDIA CUDA runtime.
Build from repository root: docker build -f gpu/Dockerfile.gql -t gql-gpu-fabric:cuda .

ARG CUDA_IMAGE=nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
FROM ${CUDA_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
PYTHONDONTWRITEBYTECODE=1
PYTHONUNBUFFERED=1
PIP_NO_CACHE_DIR=1
GQL_HOME=/opt/gql
MODEL_DIR=/models
HF_HOME=/cache/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends
python3 python3-pip ca-certificates tini
&& rm -rf /var/lib/apt/lists/*
&& groupadd --system gql
&& useradd --system --gid gql --home-dir /opt/gql --create-home gql
&& mkdir -p /opt/gql /models /cache/huggingface
&& chown -R gql:gql /opt/gql /models /cache

WORKDIR /opt/gql
COPY gpu/requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /tmp/requirements.txt
&& rm /tmp/requirements.txt
COPY --chown=gql:gql gpu/runtime /opt/gql/runtime
COPY --chown=gql:gql gpu/schema /opt/gql/schema
USER gql
EXPOSE 8080
VOLUME ["/models", "/cache"]
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python3", "-m", "uvicorn", "runtime.app:app", "--host", "0.0.0.0", "--port", "8080"]
""",
```
```"gpu/Dockerfile.cuda": r"""
ARG CUDA_IMAGE=nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
FROM ${CUDA_IMAGE}
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip ca-certificates tini
&& rm -rf /var/lib/apt/lists/*
WORKDIR /opt/gql
COPY gpu/requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt
COPY gpu/runtime ./runtime
COPY gpu/schema ./schema
RUN useradd --system --create-home gql && mkdir -p /models /cache && chown -R gql:gql /opt/gql /models /cache
USER gql
EXPOSE 8080
VOLUME ["/models", "/cache"]
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python3", "-m", "uvicorn", "runtime.app:app", "--host", "0.0.0.0", "--port", "8080"]
""",
```
```"gpu/Dockerfile.cpu": r"""
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends tini ca-certificates
&& rm -rf /var/lib/apt/lists/*
&& useradd --system --create-home gql
WORKDIR /opt/gql
COPY gpu/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt
COPY gpu/runtime ./runtime
COPY gpu/schema ./schema
RUN mkdir -p /models /cache && chown -R gql:gql /opt/gql /models /cache
USER gql
EXPOSE 8080
VOLUME ["/models", "/cache"]
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "-m", "uvicorn", "runtime.app:app", "--host", "0.0.0.0", "--port", "8080"]
""",
```
```"gpu/Dockerfile.rocm": r"""

ROCm base tags and supported devices vary by ROCm release.
Override ROCM_IMAGE with a currently supported image for your host/device.

ARG ROCM_IMAGE=rocm/dev-ubuntu-22.04
FROM ${ROCM_IMAGE}
ENV DEBIAN_FRONTEND=noninteractive PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip ca-certificates tini
&& rm -rf /var/lib/apt/lists/*
WORKDIR /opt/gql
COPY gpu/requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt
COPY gpu/runtime ./runtime
COPY gpu/schema ./schema
RUN useradd --system --create-home gql && mkdir -p /models /cache && chown -R gql:gql /opt/gql /models /cache
USER gql
EXPOSE 8080
VOLUME ["/models", "/cache"]
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python3", "-m", "uvicorn", "runtime.app:app", "--host", "0.0.0.0", "--port", "8080"]
""",
"gpu/requirements.txt": r"""
fastapi==0.115.12
uvicorn[standard]==0.34.0
pydantic==2.10.6
""",
```
```"gpu/requirements-gpu.txt": r"""

Optional framework dependencies go here, pinned to the target host/runtime.
Install the correct PyTorch CUDA/ROCm wheel separately according to official framework guidance.

-r requirements.txt
""",
```
```"gpu/runtime/init.py": "",
```
```"gpu/runtime/hardware.py": r"""
"""Hardware capability discovery without requiring a GPU framework."""

import os
import platform
import shutil
import subprocess

def _command_output(command):
executable = shutil.which(command[0])
if not executable:
return None
try:
result = subprocess.run(
[executable, *command[1:]],
capture_output=True, text=True, timeout=2, check=False
)
if result.returncode == 0:
return result.stdout.strip()[:2000]
except (OSError, subprocess.TimeoutExpired):
pass
return None

def detect_hardware():
nvidia = _command_output(["nvidia-smi", "-L"])
rocm = _command_output(["rocminfo"])
torch_info = {"available": False, "cuda_available": False, "device_count": 0}
try:
import torch # Optional dependency; not required for API startup.
torch_info = {
"available": True,
"cuda_available": bool(torch.cuda.is_available()),
"device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
"torch_version": str(torch.version),
}
except Exception:
pass

return {
    "platform": platform.platform(),
    "python": platform.python_version(),
    "backend_hint": (
        "cuda" if nvidia else "rocm" if rocm else "cpu_or_gpu_not_exposed"
    ),
    "nvidia_smi": nvidia,
    "rocminfo_detected": rocm is not None,
    "torch": torch_info,
    "model_dir": os.getenv("MODEL_DIR", "/models"),
}

""",
```
```"gpu/runtime/jobs.py": r"""
"""Small in-memory job registry. It records requests; it does not execute training."""
from datetime import datetime, timezone
from threading import Lock
from uuid import uuid4

_ALLOWED = {"inference", "training", "fine_tuning", "evaluation", "embedding"}
_jobs = []
_lock = Lock()

def create_job(kind, model, parameters=None):
if kind not in _ALLOWED:
raise ValueError(f"Unsupported job kind: {kind}")
job = {
"id": str(uuid4()),
"kind": kind,
"model": model,
"parameters": parameters or {},
"status": "accepted",
"created_at": datetime.now(timezone.utc).isoformat(),
"note": "Registered only; connect a trusted worker to execute this job.",
}
with _lock:
_jobs.append(job)
return job

def list_jobs():
with _lock:
return [dict(job) for job in _jobs]

def clear_jobs_for_tests():
with _lock:
_jobs.clear()
""",
```
```"gpu/runtime/app.py": r"""
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
return {"items": list_jobs(), "count": len(list_jobs())}

@app.post("/jobs", status_code=202)
def submit_job(request: JobRequest):
try:
return create_job(request.kind, request.model, request.parameters)
except ValueError as exc:
raise HTTPException(status_code=422, detail=str(exc)) from exc
""",
```
```"gpu/schema/schema.gql": r"""
"""GQL GPU Fabric API schema contract (GraphQL SDL)."""
type Query {
health: Health!
hardware: Hardware!
jobs: [Job!]!
}

type Mutation {
registerJob(input: RegisterJobInput!): Job!
}

type Health {
status: String!
service: String!
version: String!
}

type Hardware {
platform: String!
python: String!
backend_hint: String!
rocminfo_detected: Boolean!
model_dir: String!
}

type Job {
id: ID!
kind: String!
model: String!
status: String!
created_at: String!
note: String!
}

input RegisterJobInput {
kind: String!
model: String!
}
""",
```
```"gpu/config/runtime.gpu": r"""

YAML syntax; extension .gpu identifies a GQL runtime profile.

profile: default
api:
host: 0.0.0.0
port: 8080
authentication_required_at_gateway: true
hardware:
preferred_backend: auto
supported_backends:
- cuda
- rocm
- cpu
workloads:
enabled:
- inference
- training
- fine_tuning
- evaluation
- embedding
job_submission_executes_workload: false
storage:
model_dir: /models
cache_dir: /cache
safety:
run_as_non_root: true
expose_docker_socket: false
require_authenticated_gateway_for_network_exposure: true
""",
"gpu/environment.conda.yml": r"""
name: gql-gpu-fabric
channels:

conda-forge
dependencies:
python=3.11
pip
ca-certificates
pip:
-r requirements.txt
""",
"datacenter/compose.yaml": r"""
services:
gql-api:
build:
context: ..
dockerfile: gpu/Dockerfile.cpu
image: gql-gpu-fabric:cpu
restart: unless-stopped
ports:
"127.0.0.1:8080:8080"
environment:
MODEL_DIR: /models
HF_HOME: /cache/huggingface
volumes:
model-store:/models
model-cache:/cache
healthcheck:
test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=2)"]
interval: 30s
timeout: 5s
retries: 3
start_period: 10s
security_opt:
no-new-privileges:true
cap_drop:
ALL

volumes:
model-store:
model-cache:

For NVIDIA deployment, change dockerfile to gpu/Dockerfile.gql and configure
the Compose GPU device reservation supported by your Docker Compose version.
Keep this service bound to localhost unless an authenticated reverse proxy is configured.

""",
```
```"tests/test_jobs.py": r"""
import unittest
from runtime.jobs import clear_jobs_for_tests, create_job, list_jobs

class JobRegistryTests(unittest.TestCase):
def setUp(self):
clear_jobs_for_tests()

def tearDown(self):
    clear_jobs_for_tests()

def test_register_job(self):
    job = create_job("inference", "example/model", {"max_tokens": 16})
    self.assertEqual(job["status"], "accepted")
    self.assertEqual(job["model"], "example/model")
    self.assertEqual(len(list_jobs()), 1)

def test_reject_unknown_kind(self):
    with self.assertRaises(ValueError):
        create_job("destroy_cluster", "example/model")

def test_registry_returns_copy(self):
    create_job("evaluation", "example/model")
    records = list_jobs()
    records.clear()
    self.assertEqual(len(list_jobs()), 1)

if name == "main":
unittest.main()
""",
```
```"tests/test_hardware.py": r"""
import unittest
from runtime.hardware import detect_hardware

class HardwareTests(unittest.TestCase):
def test_detection_has_stable_contract(self):
info = detect_hardware()
self.assertIn("backend_hint", info)
self.assertIn(info["backend_hint"], {
"cuda", "rocm", "cpu_or_gpu_not_exposed"
})
self.assertIn("torch", info)
self.assertIn("cuda_available", info["torch"])

if name == "main":
unittest.main()
""",
".dockerignore": r"""
.git
.venv
pycache
*.py[cod]
.pytest_cache
.env
models/
cache/
""",
}
for rel, content in files.items():
p = root / rel
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(textwrap.dedent(content).lstrip(), encoding="utf-8")

Run syntax compilation and unit tests. Runtime tests only use standard library for imports.

env = os.environ.copy()
env["PYTHONPATH"] = str(root / "gpu") + os.pathsep + env.get("PYTHONPATH", "")
compile_result = subprocess.run(
[sys.executable, "-m", "compileall", "-q", str(root / "gpu/runtime"), str(root / "tests")],
capture_output=True, text=True, env=env
)
test_result = subprocess.run(
[sys.executable, "-m", "unittest", "discover", "-s", str(root / "tests"), "-v"],
capture_output=True, text=True, env=env
)

Package source (omit generated bytecode/cache directories).

zip_path = pathlib.Path("/mnt/data/gql_gpu_fabric.zip")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
for p in root.rglob("*"):
if p.is_file() and "pycache" not in p.parts and p.suffix != ".pyc":
zf.write(p, p.relative_to(root.parent))

print("Created:", root)
print("Files:", sum(1 for p in root.rglob("*") if p.is_file() and "pycache" not in p.parts and p.suffix != ".pyc"))
print("Compile check:", "PASS" if compile_result.returncode == 0 else "FAIL")
if compile_result.stderr:
print(compile_result.stderr)
print("Unit tests:", "PASS" if test_result.returncode == 0 else "FAIL")
print(test_result.stdout)
print(test_result.stderr)
print("ZIP:", zip_path, "exists=", zip_path.exists(), "bytes=", zip_path.stat().st_size if zip_path.exists() else 
```


The planned implementation covers:

```gpu/Dockerfile.gql — primary GPU container.

.gpu — runtime configuration.

.conda — Conda environment.

.cuda — NVIDIA CUDA runtime.

ROCm and CPU fallback images.

.gql — GraphQL schema.

datacenter/compose.yaml — deployment configuration.

Training, fine-tuning, inference, API serving, evaluation, and automated tests.
