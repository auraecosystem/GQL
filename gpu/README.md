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

## Validation limits

Unit/API tests cover the serving contract without requiring a physical GPU. Docker builds, framework initialization, model output compatibility, and real-device inference must be tested on compatible hosts. No training or fine-tuning execution is implied by this serving foundation.
