# GQL GPU Fabric multi-backend runtime

GQL provides a shared FastAPI/GraphQL control plane with separate CPU, NVIDIA CUDA, and AMD ROCm container environments.

## Start a backend

Run from the repository root:

- CPU: `docker compose -f gpu/compose.yaml up --build gql-cpu`
- NVIDIA CUDA: `docker compose -f gpu/compose.yaml --profile cuda up --build gql-cuda`
- AMD ROCm: `docker compose -f gpu/compose.yaml --profile rocm up --build gql-rocm`

Model and input directories default to `models/` and `inputs/` at the repository root. Override with `GQL_MODEL_DIR` and `GQL_INPUT_DIR`. The mounts are read-only inside the containers. Each backend has its own persistent job/cache volume. The host must be configured for the corresponding GPU runtime.

CUDA and ROCm images install shared dependencies from `gpu/requirements.txt`, then one backend-specific PyTorch set from `gpu/requirements/cuda.txt` or `gpu/requirements/rocm.txt`. These are compatibility baselines, not guarantees for every GPU, driver, or host. Existing YOLOv7 and Caffe SSD adapters use OpenCV DNN; installing PyTorch does not automatically accelerate them.

## Authenticated inference

Set a non-empty `GQL_API_TOKEN` to enable `POST /v1/inference`. Send `Authorization: Bearer <token>` with an inference request, for example:

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

The endpoint dispatches only through an explicitly enabled trusted worker. It returns 503 when disabled, 401 for invalid credentials, and 422 for invalid requests. Existing `/jobs` and GraphQL endpoints are separate and are not automatically protected by this token; use deployment-level access controls and rate limits before public exposure.

## Script analyzer agent

The static-only `script_analyzer_bot` is available behind an explicit opt-in. Set
`GQL_ENABLE_SCRIPT_ANALYZER_AGENT=1` and configure a non-empty `GQL_API_TOKEN`
before using the authenticated REST endpoint:

`POST /v1/agents/script-analyzer` with `Authorization: Bearer <token>` and JSON
body `{"scripts":{"src/main.js":"import './util.js';","src/util.js":"export {};"} }`.

Use the GraphQL mutation with variables so file names containing dots and slashes remain valid JSON keys:

```json
{
  "query": "mutation Analyze($input: AnalyzeScriptsInput!) { analyzeScripts(input: $input) }",
  "variables": {
    "input": {
      "scripts": {
        "src/main.js": "import './util.js';",
        "src/util.js": "export {};"
      }
    }
  }
}
```

Send this body to `POST /graphql`. GraphQL is an existing unauthenticated control
surface, so keep it behind trusted deployment access controls.

The agent accepts up to 100 relative JS/TS source files, with 256,000 characters per
file and 1,000,000 characters combined. It reports relative dependencies, external
imports, unresolved relative imports, cycles, and a suggested dependency-first order.
It analyzes caller-supplied source text only: it does not execute code, read arbitrary
paths, install packages, or access the network. Import parsing is heuristic and is not
a replacement for a language compiler or bundler.

## Validation limits

Unit/API tests cover the serving contract without requiring a physical GPU. Docker builds, framework initialization, model output compatibility, and real-device inference must be tested on compatible hosts. No training or fine-tuning execution is implied by this serving foundation.
