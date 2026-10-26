[![GQL GPU Fabric CI](https://github.com/auraecosystem/GQL/actions/workflows/ci.yml/badge.svg)](https://github.com/auraecosystem/GQL/actions/workflows/ci.yml)
# GQL

Let's invent it properly: `gpu/Dockerfile.gql` becomes the foundation of an intelligent, multi-backend AI execution platform for LMLM — supporting training, fine-tuning, inference, multimodal models, API serving, and evaluation.

I'll treat this as a new design rather than an existing file audit. The key architectural decision is to support multiple hardware backends through separate compatible runtime images, with one unified orchestration layer.

## 1. Invented architecture: GQL GPU Fabric

![Dell AI Factory 2.0: Packing 256 Blackwell Ultra GPUs Into One Rack | AI2Work](https://images.openai.com/static-rsc-4/NiWSzYJVDp9kTQX3287nfyeeTzlx1H8wdFCSXRW4JOLcUN02LtrWOqDcH6hOPbITT3rQ6jNSy0vk7ZO5FXS6rtUYKxDwmXWbzzZqYlpNIJLTU6GMCz1aybdDeFVKGLpA8SNBx23KNUBbfc4KmITcdvtEkXapYj2CXHh4Xg0TmiU?purpose=inline)

Add to Favorites

## GQL — General GPU Layer

An invented project name, not an established industry standard.

The platform has six cooperating subsystems:

GQL Orchestrator

Job scheduling · workload routing · resource policy

Compute adapters

CUDA · ROCm · CPU

Model runtime

LMLM · PyTorch · Transformers

Artifact storage

Weights · checkpoints · datasets

Observability

Metrics · logs · benchmarks

Workload types: training, fine-tuning, inference, multimodal generation, API serving, and evaluation.

## 2. Invent the Dockerfile

Here is a starting implementation for `gpu/Dockerfile.gql`. It builds an NVIDIA CUDA-based Python runtime. It is a foundation for the platform, not yet a complete model server.

dockerfile

```Dockerfile
# gpu/Dockerfile.gql
# GQL — General GPU Layer
# NVIDIA CUDA runtime foundation for LMLM workloads.

ARG CUDA_IMAGE=nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
FROM ${CUDA_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    GQL_HOME=/opt/gql \
    MODEL_DIR=/models \
    HF_HOME=/cache/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    ca-certificates \
    tini \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --system gql \
    && useradd --system --gid gql \
       --home-dir /opt/gql --create-home gql \
    && mkdir -p /opt/gql /models /cache/huggingface \
    && chown -R gql:gql /opt/gql /models /cache

WORKDIR /opt/gql

# Install pinned dependencies from the build context.
# Provide gpu/requirements-gpu.txt before building.
COPY requirements-gpu.txt /tmp/requirements-gpu.txt

RUN python3 -m pip install --no-cache-dir \
      --requirement /tmp/requirements-gpu.txt \
    && rm /tmp/requirements-gpu.txt

COPY --chown=gql:gql . /opt/gql

USER gql

VOLUME ["/models", "/cache"]

# The application must implement this module before serving works.
EXPOSE 8080

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python3", "-m", "lmlm"]
```

### What this gives us

* A CUDA-compatible runtime foundation for NVIDIA workloads.

* Non-root execution and persistent model/cache directories.

* A configurable application image for LMLM components.

* A clear extension point for the inference server and training workers.

Important: The image is not validated or built here. It requires a compatible `requirements-gpu.txt`, an actual `lmlm` Python module, and a compatible host GPU driver. The selected CUDA image is an example baseline, not a guarantee that every model or GPU supports it.

## 3. Expand to all hardware

Use separate images with a shared application contract.

| Runtime     | Purpose                  | Implementation                           |
| ----------- | ------------------------ | ---------------------------------------- |
| `gql-cuda`  | NVIDIA GPUs              | CUDA-compatible base image               |
| `gql-rocm`  | AMD GPUs                 | Compatible ROCm base image               |
| `gql-cpu`   | CPU execution            | CPU-compatible Python image              |
| `gql-train` | Training and fine-tuning | Framework-specific training dependencies |
| `gql-serve` | Model inference and APIs | Minimal serving dependencies             |
| `gql-eval`  | Benchmarks and testing   | Evaluation tools and metrics             |

The CUDA and ROCm environments must remain separate because their device runtimes and framework builds are not interchangeable. CPU execution should be a supported alternative, not a promise of equivalent performance.

## 4. Proposed repository layout

```bash
gpu/
├── Dockerfile.gql
├── Dockerfile.cuda
├── Dockerfile.rocm
├── Dockerfile.cpu
├── Dockerfile.train
├── Dockerfile.serve
├── Dockerfile.eval
├── compose.yaml
├── requirements/
│   ├── common.txt
│   ├── cuda.txt
│   ├── rocm.txt
│   └── training.txt
├── gql/
│   ├── orchestrator.py
│   ├── hardware.py
│   ├── model_registry.py
│   ├── job_scheduler.py
│   ├── inference.py
│   ├── training.py
│   ├── evaluation.py
│   └── health.py
├── tests/
│   ├── test_hardware.py
│   ├── test_model_loading.py
│   ├── test_inference.py
│   └── test_job_scheduler.py
└── README.md
```

This is a proposed structure; the additional files and modules do not exist until implemented.

## 5. Execution roadmap

1. Foundation: Build the CUDA, ROCm, and CPU images; pin dependencies and verify hardware detection.

2. Model runtime: Implement model loading, memory checks, inference, and model-specific adapters.

3. Training: Add training jobs, checkpoint persistence, resumable execution, and fine-tuning.

4. Serving: Add authenticated APIs, request limits, health checks, and graceful shutdown.

5. Validation: Run unit tests, container builds, inference smoke tests, and GPU benchmarks on actual compatible hardware.

My recommendation is to make GQL the compute and container infrastructure beneath LMLM, not another competing model. LMLM remains the intelligence and model-runtime ecosystem; GQL manages the environments and execution resources those models need.

The next implementation milestone is the working multi-backend runtime, including its dependency files, Compose configuration, model-serving API, and automated tests.


## YOLOv7 ONNX inference worker

The YOLOv7 adapter is opt-in and handles `inference` jobs using an ONNX model already present in `MODEL_DIR`. It does not download models, accept uploaded executable artifacts, or read image paths outside `GQL_INPUT_DIR`.

Configure the service with `GQL_ENABLE_YOLOV7_WORKER=1`, mount the approved ONNX model into `/models`, and mount input images into `/inputs`. Submit a job with `kind: "inference"`, `model: "yolov7.onnx"`, and parameters such as `{"input_path":"frame.jpg","backend":"cpu","image_size":640}`; then dispatch its returned job ID through `POST /jobs/{job_id}/dispatch`.

Supported image extensions: JPG, JPEG, PNG, WEBP, BMP. `backend` accepts `auto`, `cpu`, or `cuda`; `auto` intentionally uses CPU for predictable compatibility. Explicit CUDA requires an OpenCV build with CUDA DNN support and a compatible host runtime. The detector currently expects decoded YOLO-style rows shaped like `[x, y, width, height, objectness, class scores...]`; validate output semantics against the exact exported ONNX model before production use.

Example container environment: `GQL_ENABLE_YOLOV7_WORKER=1`. Model and input files must be provided through trusted mounted directories. The worker does not make training, fine-tuning, embedding, or evaluation jobs executable.


## Caffe SSD inference worker

GQL also has an opt-in Caffe SSD adapter for models exported with a Caffe `deploy.prototxt` and trained `.caffemodel` weights. Enable it with `GQL_ENABLE_CAFFE_SSD_WORKER=1`; mount both trusted model files in `/models` and images in `/inputs`.

Submit an inference job with `model: "deploy.prototxt"` and parameters such as `{"weights_path":"trained.caffemodel","input_path":"frame.jpg","confidence_threshold":0.25}`, then dispatch the job normally. Preprocessing uses a 300×300 warped BGR image and mean subtraction (104, 117, 123), matching the supplied training configuration.

**Important limitations:** this adapter expects the deploy network to emit the common Caffe `DetectionOutput` tensor with rows `[image_id, class_id, confidence, xmin, ymin, xmax, ymax]` and normalized coordinates. The training prototxt alone is not a deploy model and does not contain trained weights. If the model emits raw localization, confidence, and prior-box tensors instead, a matching SSD decoder is required. The adapter currently uses CPU OpenCV DNN; real model inference and output compatibility must be verified with the actual deploy files before production use.


## Multi-backend runtime foundation

The Compose configuration provides separate CPU, NVIDIA CUDA, and AMD ROCm services:

- CPU: `docker compose -f gpu/compose.yaml up --build gql-cpu`
- NVIDIA: `docker compose -f gpu/compose.yaml --profile cuda up --build gql-cuda`
- AMD ROCm: `docker compose -f gpu/compose.yaml --profile rocm up --build gql-rocm`

Default model and input bind mounts resolve from the repository root (`models/` and `inputs/`). Override them with `GQL_MODEL_DIR` and `GQL_INPUT_DIR`. Model and input mounts are read-only in containers; each backend has a separate persistent job/cache volume. The target host must provide the corresponding GPU runtime and device access.

CUDA and ROCm Dockerfiles install the shared API/OpenCV dependencies first, then a separate pinned PyTorch wheel set from `gpu/requirements/cuda.txt` or `gpu/requirements/rocm.txt`. These pins are a compatibility baseline, not proof that a specific GPU/driver combination works. Existing YOLOv7 and Caffe SSD workers use OpenCV DNN; installing PyTorch does not automatically accelerate them.

### Synchronous inference API

Set a non-empty `GQL_API_TOKEN` to enable `POST /v1/inference`. Send `Authorization: Bearer <token>` and a JSON body such as:

```json
{
  "kind": "inference",
  "model": "yolov7.onnx",
  "parameters": {
    "input_path": "frame.jpg",
    "backend": "cpu",
    "image_size": 640
  }
}
```

The endpoint dispatches only through an explicitly enabled trusted inference worker and returns the completed job result. It returns 503 if the token is not configured or no worker is available, 401 for missing/invalid credentials, and 422 for invalid requests. Existing `/jobs` and GraphQL control-plane endpoints remain separate; protect the whole service with deployment-level access controls and rate limits before exposing it publicly.

### Verification boundaries

Unit/API tests cover token handling and trusted-worker dispatch without requiring a physical GPU. Docker image builds, CUDA/ROCm framework initialization, model-specific output compatibility, and real-device inference still need validation on compatible hosts; they are not implied by unit tests.
