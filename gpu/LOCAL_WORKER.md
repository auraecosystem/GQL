# Opt-in local Transformers worker

The local text-generation worker is disabled by default.

1. Install a PyTorch build compatible with the host CPU/CUDA/ROCm runtime.
2. Install `gpu/requirements-inference.txt`.
3. Place reviewed model and tokenizer artifacts under `MODEL_DIR`. The model directory must include `config.json`; GQL does not download model files.
4. Set `GQL_ENABLE_LOCAL_TRANSFORMERS_WORKER=1` and restart the service.

The adapter only registers for `inference`. It uses `local_files_only=True` and
`trust_remote_code=False`, rejects paths outside `MODEL_DIR`, limits prompts to 20,000
characters and output to 512 tokens. It is not a sandbox for malicious model artifacts.
Training, fine-tuning, embeddings and evaluation remain unimplemented by this adapter.

Submit a job with `kind: "inference"`, `model: "my-local-model"` (relative to
`MODEL_DIR`), and `parameters: {"prompt": "Hello", "max_new_tokens": 80}`, then call
`POST /jobs/{job_id}/dispatch`. Registering a job does not start execution.
