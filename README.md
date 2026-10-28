# GQL GPU Fabric

[![GQL GPU Fabric CI](https://github.com/auraecosystem/GQL/actions/workflows/ci.yml/badge.svg)](https://github.com/auraecosystem/GQL/actions/workflows/ci.yml)

**GQL GPU Fabric** is a lightweight AI workload control plane that exposes HTTP and GraphQL interfaces for hardware inspection, persistent job registration, and controlled dispatch to explicitly enabled local inference workers.

The project is designed to grow into a multi-backend execution layer for CPU, NVIDIA CUDA, and AMD ROCm environments. Its current implementation focuses on the service API, job lifecycle, hardware reporting, container entry points, and opt-in inference adapters—not a universal training platform.

> **Status matters:** a job kind being accepted by the registry does not mean an executor exists for it. Training, fine-tuning, embedding, and evaluation are represented in the job contract, but they are not currently implemented as runnable workloads by the included inference workers.

## Contents

- [Project goals](#project-goals)
- [Current capabilities](#current-capabilities)
- [Architecture](#architecture)
- [Repository layout](#repository-layout)
- [Requirements](#requirements)
- [Quick start](#quick-start)
- [Docker Compose](#docker-compose)
- [HTTP API](#http-api)
- [GraphQL API](#graphql-api)
- [Jobs and lifecycle](#jobs-and-lifecycle)
- [Inference workers](#inference-workers)
- [Configuration](#configuration)
- [Storage and persistence](#storage-and-persistence)
- [Security model](#security-model)
- [Development and tests](#development-and-tests)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)

## Project goals

GQL is intended to provide a consistent control surface for AI workloads across different compute environments.

- **One control plane:** common health, hardware, job, and inference interfaces.
- **Explicit worker registration:** workloads run only when a compatible worker is enabled and registered.
- **Persistent job state:** SQLite-backed job records survive process restarts when the database is placed on persistent storage.
- **Backend flexibility:** separate CPU, CUDA, and ROCm container environments, because their drivers and framework builds are not interchangeable.
- **Local model custody:** model files are supplied by the operator through mounted directories rather than downloaded automatically by inference adapters.
- **Incremental evolution:** add worker types and model runtimes behind a stable job contract without implying that every workload is already supported.

GQL is an execution and orchestration layer, not itself a foundation model. It can host or dispatch to model runtimes when the appropriate adapter, dependencies, model artifacts, and hardware are available.

## Current capabilities

| Capability | Current status | Notes |
|---|---|---|
| FastAPI service | Implemented | Service entry point is `runtime.app:app`. |
| Health endpoint | Implemented | HTTP `GET /health`; GraphQL `Query.health`. |
| Hardware report | Implemented | HTTP `GET /hardware`; GraphQL `Query.hardware`. Hardware visibility depends on the host and container runtime. |
| GraphQL over HTTP | Implemented | JSON POST to `/graphql`. |
| Persistent job registry | Implemented | SQLite-backed job creation, listing, retrieval, and guarded lifecycle transitions. |
| Job dispatch | Implemented | Dispatches through a registered trusted worker; unsupported jobs are not executed. |
| Token-protected synchronous inference route | Implemented | `POST /v1/inference` requires `GQL_API_TOKEN` and a registered inference worker. |
| YOLOv7 ONNX inference | Opt-in adapter | Requires an approved local ONNX model and input images. Verify model output semantics before production use. |
| Caffe SSD inference | Opt-in adapter | Requires a compatible deploy prototxt and trained caffemodel; CPU OpenCV DNN path. |
| Local Transformers text generation | Opt-in adapter | Disabled by default; requires the adapter's compatible dependencies and a local model. |
| CPU container | Available in Compose configuration | Suitable for CPU-compatible workers and API development. |
| CUDA container | Available in Compose configuration | NVIDIA runtime; actual GPU access depends on host drivers and Docker GPU configuration. |
| ROCm container | Available in Compose configuration | AMD runtime; actual hardware support depends on the host, device access, and compatible packages. |
| Training and fine-tuning | Not implemented as runnable workers | Job kinds can be registered, but no included training executor makes them run. |
| Embeddings and evaluation | Not implemented as runnable workers | These job kinds are not made executable by registration alone. |
| General multimodal generation | Planned | Requires concrete model adapters, contracts, dependencies, and tests. |
| Distributed scheduling / multi-node orchestration | Planned | Not provided by the current SQLite-backed service. |

## Architecture

```text
Clients / automation
        |
        | HTTP JSON / GraphQL
        v
+---------------------------+
| FastAPI service           |
| /health  /hardware        |
| /jobs    /graphql         |
| /v1/inference             |
+-------------+-------------+
              |
      +-------+--------+
      |                |
      v                v
 GraphQL schema    Job registry
                   SQLite state
                   accepted/running/
                   completed/failed
                         |
                         v
                 Trusted worker registry
                  /        |        \
                 v         v         v
              YOLOv7    Caffe SSD  Local Transformers
              ONNX      inference  text generation
                         |
                         v
               Local model/input mounts
                         |
             CPU / CUDA-capable runtime
```

The service is packaged for multiple hardware environments, but each worker still has its own compatibility requirements. A CUDA-enabled container does not guarantee that a GPU is visible, and a ROCm image is not interchangeable with a CUDA image.

### Main components

- **API layer:** FastAPI routes for service health, hardware reporting, job management, dispatch, and token-gated inference.
- **GraphQL layer:** schema-backed queries and mutations served at `POST /graphql`.
- **Hardware detection:** reports host/runtime signals such as Python, platform, NVIDIA utilities, ROCm detection, and PyTorch device availability where installed.
- **Job registry:** SQLite persistence with atomic claiming and expected-status transitions to reduce duplicate dispatch.
- **Worker registry:** routes a job to an explicitly registered adapter. If no compatible worker is registered, dispatch returns an unavailable-worker error rather than pretending the workload ran.
- **Container profiles:** CPU, CUDA, and ROCm configurations defined in `gpu/compose.yaml`.

## Repository layout

```text
.
├── .github/
│   └── workflows/
│       └── ci.yml
├── gpu/
│   ├── Dockerfile.cpu
│   ├── Dockerfile.gql
│   ├── Dockerfile.rocm
│   ├── compose.yaml
│   ├── requirements.txt
│   ├── requirements-dev.txt
│   ├── requirements/
│   │   └── cuda.txt
│   ├── runtime/
│   │   ├── app.py
│   │   ├── hardware.py
│   │   ├── jobs.py
│   │   ├── workers.py
│   │   ├── yolov7.py
│   │   ├── caffe_ssd.py
│   │   └── local_worker.py
│   └── schema/
│       └── schema.gql
└── tests/
```

This is a guide to the principal implementation files; consult the repository tree for the authoritative current list.

## Requirements

For local development:

- Python 3.11 or 3.12
- pip
- Git
- A compatible OS for the selected dependencies

For container use:

- Docker Engine and the Docker Compose plugin
- For NVIDIA acceleration: compatible host NVIDIA drivers and NVIDIA Container Toolkit
- For AMD acceleration: a compatible ROCm host stack, device permissions, and supported hardware
- Approved model files and input data for whichever worker you enable

GPU support is not supplied by installing Python packages alone. The host driver, container runtime, framework build, GPU architecture, and model requirements must agree.

## Quick start

### 1. Clone the repository

```bash
git clone https://github.com/auraecosystem/GQL.git
cd GQL
```

### 2. Create a virtual environment

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r gpu/requirements.txt
```

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r gpu/requirements.txt
```

### 3. Start the API

The runtime package is under `gpu/runtime`, so set `PYTHONPATH=gpu`.

Linux/macOS:

```bash
PYTHONPATH=gpu python -m uvicorn runtime.app:app --host 127.0.0.1 --port 8080
```

Windows PowerShell:

```powershell
$env:PYTHONPATH = "gpu"
python -m uvicorn runtime.app:app --host 127.0.0.1 --port 8080
```

The service listens on `http://127.0.0.1:8080`. Keeping the development server bound to loopback avoids exposing it to the network by default.

### 4. Check service health

```bash
curl http://127.0.0.1:8080/health
curl http://127.0.0.1:8080/hardware
```

Interactive OpenAPI documentation is available at `/docs` while the service is running.

## Docker Compose

Compose configuration lives at `gpu/compose.yaml`. Run commands from the repository root.

### CPU profile

```bash
docker compose -f gpu/compose.yaml up --build gql-cpu
```

The CPU service is published on port `8080` by default.

### NVIDIA CUDA profile

First confirm that Docker can see the host GPU. Then run:

```bash
docker compose -f gpu/compose.yaml --profile cuda up --build gql-cuda
```

The CUDA service is published on port `8081` by default. The image uses the configurable `GQL_CUDA_IMAGE` build argument, defaulting to a CUDA 12.4.1 cuDNN runtime base. Framework and driver compatibility must be checked for the target system.

### AMD ROCm profile

On a compatible AMD Linux host:

```bash
docker compose -f gpu/compose.yaml --profile rocm up --build gql-rocm
```

The ROCm service is published on port `8082` by default. Device nodes and group permissions are configured in Compose; the host must still provide a compatible ROCm installation and supported hardware.

### Stop services

```bash
docker compose -f gpu/compose.yaml down
```

Compose uses separate cache volumes for CPU, CUDA, and ROCm services. Model and input directories are mounted read-only by default. Configure their host paths with `GQL_MODEL_DIR` and `GQL_INPUT_DIR`.

## HTTP API

### Health

`GET /health`

Example:

```bash
curl http://127.0.0.1:8080/health
```

Returns service status, service identifier, and version.

### Hardware

`GET /hardware`

Returns the hardware-detection report. Treat this as diagnostic information, not a guarantee that every detected device can run every model.

### Register a job

`POST /jobs`

```bash
curl -X POST http://127.0.0.1:8080/jobs \
  -H 'Content-Type: application/json' \
  -d '{
    "kind": "inference",
    "model": "yolov7.onnx",
    "parameters": {
      "input_path": "frame.jpg",
      "backend": "cpu",
      "image_size": 640
    }
  }'
```

Supported job-kind labels are `inference`, `training`, `fine_tuning`, `evaluation`, and `embedding`. These are registry labels, not a promise that each kind has an executable worker. The response includes a job ID and initial status, normally `accepted`.

### List jobs

`GET /jobs`

Returns the current retained job records and a count.

### Retrieve a job

`GET /jobs/{job_id}`

Returns a job record or `404` if the ID is unknown or no longer retained.

### Dispatch a job

`POST /jobs/{job_id}/dispatch`

```bash
curl -X POST http://127.0.0.1:8080/jobs/JOB_ID/dispatch
```

Dispatch claims an accepted job, invokes the registered trusted worker, and records the result or failure. A job that cannot be claimed because it is already running or terminal returns `409`. If no worker can execute it, the API returns `503` and restores the job to `accepted` with a note. Worker execution failures return a generic API error and mark the job failed.

### Authenticated inference

`POST /v1/inference` is a synchronous inference route. It is disabled until the operator sets `GQL_API_TOKEN`.

```bash
export GQL_API_TOKEN='replace-with-a-long-random-secret'

curl -X POST http://127.0.0.1:8080/v1/inference \
  -H "Authorization: Bearer $GQL_API_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{
    "kind": "inference",
    "model": "yolov7.onnx",
    "parameters": {
      "input_path": "frame.jpg",
      "backend": "cpu",
      "image_size": 640
    }
  }'
```

The endpoint accepts only `kind: "inference"`. A configured token does not install or enable a worker by itself; the selected worker must also be explicitly enabled and its model artifacts must be available.

## GraphQL API

Send GraphQL requests as JSON to `POST /graphql`. The payload accepts `query`, optional `variables`, and optional `operationName`. The service rejects missing or empty query strings, queries larger than 100,000 characters, and non-object variables.

### Health query

```bash
curl -X POST http://127.0.0.1:8080/graphql \
  -H 'Content-Type: application/json' \
  -d '{"query":"query { health { status service version } }"}'
```

### Hardware query

```json
{
  "query": "query { hardware { platform python backend_hint rocminfo_detected model_dir torch { available cuda_available device_count version } } }"
}
```

### List jobs

```json
{
  "query": "query { jobs { id kind model status created_at note } }"
}
```

### Register a job

```json
{
  "query": "mutation Register($input: RegisterJobInput!) { registerJob(input: $input) { id kind model status created_at note } }",
  "variables": {
    "input": {
      "kind": "inference",
      "model": "yolov7.onnx",
      "parameters": {
        "input_path": "frame.jpg",
        "backend": "cpu",
        "image_size": 640
      }
    }
  }
}
```

Job registration records intent; it does **not** run inference. Dispatch is exposed through the HTTP endpoint `POST /jobs/{job_id}/dispatch`. Consult `gpu/schema/schema.gql` for the exact schema types and fields for the checked-out revision.

GraphQL execution is synchronous. Do not expose this service publicly without adding and validating appropriate authentication, authorization, request limits, and operational controls for your deployment.

## Jobs and lifecycle

The current registry stores job records in SQLite. Typical lifecycle:

```text
accepted --dispatch/claim--> running --worker succeeds--> completed
                                  |
                                  +--worker fails-------> failed
```

If a worker is unavailable, dispatch records a note and returns the job to `accepted`, allowing an operator to enable a compatible worker and retry. A job must be in `accepted` state to be claimed. Atomic status transitions help prevent concurrent requests from executing the same accepted job at once.

The registry retains up to 1,000 records, removing older entries as new jobs are created. This is a lightweight local registry, not a distributed queue: there is no multi-node scheduler, lease renewal, worker heartbeat, or automatic retry policy.

## Inference workers

All workers are opt-in. Mount model artifacts and input data from trusted sources, and validate a model against its adapter before using it with important workloads.

### YOLOv7 ONNX

Enable with:

```bash
GQL_ENABLE_YOLOV7_WORKER=1
```

The worker expects an ONNX model in `MODEL_DIR` (normally mounted at `/models`) and images under `GQL_INPUT_DIR` (normally `/inputs`). Example job parameters:

```json
{
  "input_path": "frame.jpg",
  "backend": "cpu",
  "image_size": 640
}
```

Supported image extensions: JPG, JPEG, PNG, WEBP, BMP. The backend parameter accepts `auto`, `cpu`, or `cuda`. `auto` intentionally selects CPU for predictable compatibility. Explicit CUDA requires an OpenCV build with CUDA DNN support and a compatible runtime.

The current post-processing expects decoded YOLO-style rows shaped like `[x, y, width, height, objectness, class scores...]`. Output layout and score semantics vary between ONNX exports; verify preprocessing, output tensor layout, class mapping, and coordinate scaling for the exact model before production use.

### Caffe SSD

Enable with:

```bash
GQL_ENABLE_CAFFE_SSD_WORKER=1
```

Mount a compatible Caffe deploy network and trained weights under `/models`. Example parameters:

```json
{
  "weights_path": "trained.caffemodel",
  "input_path": "frame.jpg",
  "confidence_threshold": 0.25
}
```

The adapter expects the deploy network to emit a common Caffe `DetectionOutput` tensor with rows `[image_id, class_id, confidence, xmin, ymin, xmax, ymax]` and normalized coordinates. Preprocessing uses a 300×300 warped BGR image and mean subtraction (104, 117, 123). The training prototxt alone is not a deploy network and does not contain trained weights. Models that emit raw localization, confidence, and prior-box tensors need a matching SSD decoder.

This adapter uses CPU OpenCV DNN. Verify output compatibility against the actual deploy prototxt and weights before relying on results.

### Local Transformers worker

Enable with:

```bash
GQL_ENABLE_LOCAL_TRANSFORMERS_WORKER=1
```

This worker is intended for local text generation using model artifacts available in the configured model directory. It is disabled by default. Its supported model format, memory needs, tokenizer files, and dependency compatibility must be checked against the actual model. Keep remote model downloads and untrusted model code disabled in production. Enabling the worker does not imply that all Transformers architectures or quantization formats are supported.

### Worker selection

The YOLOv7 and Caffe SSD adapters both handle inference jobs and are registered only when explicitly enabled. If multiple inference adapters are enabled, the Caffe SSD adapter takes precedence in the current application initialization. Configure only the adapter you intend to use unless you have verified the registration behavior for your deployment.

## Configuration

The following environment variables are used by the current runtime and Compose configuration.

| Variable | Default / behavior | Purpose |
|---|---|---|
| `MODEL_DIR` | `/models` in containers | Root directory for approved model artifacts. |
| `GQL_INPUT_DIR` | `/inputs` in containers | Root directory for approved input files. |
| `GQL_JOB_DB` | `/tmp/gql-gpu-fabric-jobs.sqlite3` outside Compose | SQLite database path. Use a persistent volume for durable container state. |
| `GQL_API_TOKEN` | Empty | Enables `POST /v1/inference` when set; requests require a matching Bearer token. |
| `GQL_ENABLE_YOLOV7_WORKER` | Disabled | Explicitly enables YOLOv7 ONNX inference worker. |
| `GQL_ENABLE_CAFFE_SSD_WORKER` | Disabled | Explicitly enables Caffe SSD inference worker. |
| `GQL_ENABLE_LOCAL_TRANSFORMERS_WORKER` | Disabled | Explicitly enables the local Transformers worker. |
| `HF_HOME` | `/cache/huggingface` in containers | Hugging Face cache location for compatible local runtime use. |
| `GQL_CPU_PORT` | `8080` | Host port for CPU Compose service. |
| `GQL_CUDA_PORT` | `8081` | Host port for CUDA Compose service. |
| `GQL_ROCM_PORT` | `8082` | Host port for ROCm Compose service. |
| `GQL_MODEL_DIR` | `../models` relative to Compose project | Host model directory mounted read-only. |
| `GQL_INPUT_DIR` | `../inputs` relative to Compose project | Host input directory mounted read-only. |
| `GQL_CUDA_IMAGE` | CUDA 12.4.1 cuDNN runtime image | Overrides the CUDA base image. |
| `GQL_ROCM_IMAGE` | `rocm/dev-ubuntu-22.04` | Overrides the ROCm base image. |

Set secrets in the deployment environment or a secret manager. Do not commit tokens or other credentials to the repository.

## Storage and persistence

- **Jobs:** SQLite. In Compose, `GQL_JOB_DB` points into `/cache`, which is backed by a named volume.
- **Models:** supplied through `MODEL_DIR`; Compose mounts the host model directory read-only by default.
- **Inputs:** supplied through `GQL_INPUT_DIR`; Compose mounts the host input directory read-only by default.
- **Cache:** `HF_HOME` points under `/cache` in containers.

Back up persistent volumes if job history or cached artifacts matter. The current job database is a bounded local registry, not an audit-grade event store.

## Security model

GQL includes useful guardrails, but it is not a complete public multi-tenant service. Review the following before deployment:

1. **Network exposure:** bind local development to loopback. Put production deployments behind a properly configured gateway or private network.
2. **Authentication:** `GQL_API_TOKEN` protects `POST /v1/inference` only. It does not automatically protect the other HTTP routes or GraphQL endpoint.
3. **Authorization:** there is no general user/tenant permission model in the current service.
4. **Model custody:** provide trusted model files through read-only mounts. Do not accept arbitrary executable model artifacts from untrusted clients.
5. **Worker opt-in:** leave inference workers disabled unless needed; enable only the adapter you have reviewed and tested.
6. **Secrets:** use strong random bearer tokens and a secret manager or environment injection. Never commit secrets.
7. **Resource controls:** set container CPU, memory, GPU, request, and filesystem limits appropriate to your environment. Avoid exposing expensive inference to unrestricted callers.
8. **GraphQL controls:** query length is limited, but query depth/complexity controls and authentication should be evaluated before public exposure.
9. **Error handling:** worker internals are not intentionally returned in dispatch error responses; keep logs private and scrub sensitive data.
10. **Supply chain:** pin and review dependencies and base images, scan images, and rebuild them when security updates are required.

Do not treat the bearer token as a substitute for TLS. Use HTTPS wherever requests traverse an untrusted network.

## Development and tests

Install the development dependencies:

```bash
python -m pip install -r gpu/requirements-dev.txt
```

Run the test suite from the repository root:

Linux/macOS:

```bash
PYTHONPATH=gpu python -m compileall -q gpu/runtime tests
PYTHONPATH=gpu python -m unittest discover -s tests -v
```

Windows PowerShell:

```powershell
$env:PYTHONPATH = "gpu"
python -m compileall -q gpu/runtime tests
python -m unittest discover -s tests -v
```

The GitHub Actions workflow runs the Python test matrix on Python 3.11 and 3.12, compiles the runtime and tests, and executes the unit/API contract tests. CI unit tests do not by themselves prove that a physical GPU is available or that a specific model produces correct predictions.

Before a release, also validate:

- Container builds for each advertised profile
- Health and hardware endpoints in the built images
- Persistence across container restarts
- Token rejection and acceptance behavior for `/v1/inference`
- Disabled-worker and unknown-model failure paths
- Real inference using the exact approved model and representative input data
- GPU visibility and memory behavior on supported physical hardware
- Container permissions, read-only model/input mounts, and secret handling

Report the actual commands and results for each validation. Do not claim GPU inference or model compatibility based only on a successful unit-test run.

## Troubleshooting

### The service starts, but dispatch returns `503`

No compatible worker may be registered. Confirm that the intended worker's environment flag is enabled, its dependencies are installed, and the model/input files exist in the configured directories. Job registration alone does not execute a workload.

### `POST /v1/inference` returns `503`

Set `GQL_API_TOKEN` in the service environment. Then confirm that an inference worker is enabled. A token enables the route's authentication gate; it does not create a worker.

### `POST /v1/inference` returns `401`

Send the configured token as `Authorization: Bearer YOUR_TOKEN`. Check for whitespace or mismatched environment values without printing the secret into logs.

### CUDA is not detected

Check host drivers, NVIDIA Container Toolkit, Docker GPU access, the selected image and PyTorch build, and the output of `GET /hardware`. CUDA inside the container cannot replace a compatible host driver.

### ROCm is not detected

Check supported AMD hardware, host ROCm compatibility, device nodes (`/dev/kfd` and `/dev/dri`), group permissions, and the selected framework build.

### YOLOv7 produces incorrect detections

ONNX exports differ. Verify input size, color order, normalization, output tensor layout, objectness/class-score interpretation, coordinate conversion, and class labels against the exact export. A model loading successfully does not establish prediction correctness.

### Caffe SSD model fails to load or returns unexpected results

Use a deploy prototxt and its matching trained weights. Confirm the network emits the expected DetectionOutput format. Networks that expose raw SSD tensors require a decoder that matches their priors and training/export configuration.

### Job history disappears after container recreation

Confirm that `GQL_JOB_DB` points to a path inside a persistent volume and that the service user can write to it. The default `/tmp` database path is not intended for durable container storage.

## Roadmap

The roadmap is aspirational; items below are not claims of current functionality.

1. **Stabilize the core:** maintain the HTTP/GraphQL contracts, job lifecycle tests, and clear worker error states.
2. **Harden deployment:** authentication and authorization across all routes, GraphQL complexity controls, structured audit logs, and explicit resource limits.
3. **Validate backends:** reproducible CPU/CUDA/ROCm image builds and hardware-specific smoke tests.
4. **Formalize worker contracts:** typed inputs/outputs, capability discovery, model metadata, and versioned adapter interfaces.
5. **Add real workload executors:** implement training, fine-tuning, embedding, and evaluation as separate tested workers before advertising them as supported.
6. **Improve scheduling:** retries, cancellation, timeouts, worker health, concurrency controls, and queue back-pressure.
7. **Improve observability:** structured logs, latency and resource metrics, and traceable job events.
8. **Expand model support deliberately:** add multimodal and additional inference adapters only with documented artifacts, preprocessing, licensing, resource needs, and tests.

## Contributing

Contributions should preserve the existing API and worker behavior unless a change explicitly updates the contract and tests.

1. Open an issue describing the use case and expected behavior.
2. Keep backend-specific dependencies isolated where possible.
3. Add tests for success paths, invalid inputs, unavailable workers, and failure transitions.
4. Document required model artifacts, hardware, environment variables, and security implications.
5. Run the test suite before submitting a pull request.
6. State which checks were actually run and identify any hardware validation that could not be performed.

Security-sensitive reports should be sent privately to the repository maintainers rather than posted with exploitable details in a public issue.

## License

See the repository's license file for the applicable terms. If no license file is present in the checked-out revision, do not assume that public availability grants permission to redistribute or use the code.

---

**Project:** [auraecosystem/GQL](https://github.com/auraecosystem/GQL)  
**CI workflow:** [GQL GPU Fabric CI](https://github.com/auraecosystem/GQL/actions/workflows/ci.yml)
Example container environment: `GQL_ENABLE_YOLOV7_WORKER=1`. Model and input files must be provided through trusted mounted directories. The worker does not make training, fine-tuning, embedding, or evaluation jobs executable.
