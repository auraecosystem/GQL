"""Opt-in local Transformers inference worker; local artifacts only."""
from __future__ import annotations
import os
from pathlib import Path
from threading import RLock
from typing import Any
from runtime.workers import WorkerUnavailable

class LocalTransformersWorker:
    """Local-only causal language model inference adapter."""
    _load_lock = RLock()

    def __init__(self, model_dir: str | None = None) -> None:
        self.model_dir = Path(model_dir or os.getenv("MODEL_DIR", "/models")).expanduser().resolve()
        self._loaded: dict[str, tuple[Any, Any, Any]] = {}

    def _resolve_model_path(self, model: Any) -> Path:
        if not isinstance(model, str) or not model.strip():
            raise WorkerUnavailable("A local model directory name is required.")
        candidate = (self.model_dir / model).resolve()
        try:
            candidate.relative_to(self.model_dir)
        except ValueError as exc:
            raise WorkerUnavailable("Model path must stay inside MODEL_DIR.") from exc
        if not candidate.is_dir():
            raise WorkerUnavailable("The requested local model directory was not found.")
        if not (candidate / "config.json").is_file():
            raise WorkerUnavailable("The local model directory is missing config.json.")
        return candidate

    def _load(self, path: Path):
        key = str(path)
        with self._load_lock:
            if key in self._loaded:
                return self._loaded[key]
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer
            except ImportError as exc:
                raise WorkerUnavailable("Inference dependencies are missing. Install gpu/requirements-inference.txt and a compatible PyTorch build.") from exc
            try:
                tokenizer = AutoTokenizer.from_pretrained(key, local_files_only=True, trust_remote_code=False)
                model = AutoModelForCausalLM.from_pretrained(key, local_files_only=True, trust_remote_code=False)
                model.to("cuda" if torch.cuda.is_available() else "cpu")
                model.eval()
            except Exception as exc:
                raise WorkerUnavailable("The local model could not be loaded.") from exc
            self._loaded[key] = (torch, tokenizer, model)
            return self._loaded[key]

    def execute(self, job: dict[str, Any]) -> dict[str, Any]:
        if job.get("kind") != "inference":
            raise WorkerUnavailable("The local Transformers worker supports inference jobs only.")
        parameters = job.get("parameters") or {}
        if not isinstance(parameters, dict):
            raise WorkerUnavailable("Inference parameters must be a JSON object.")
        prompt = parameters.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise WorkerUnavailable("Inference requires a non-empty 'prompt' parameter.")
        if len(prompt) > 20_000:
            raise WorkerUnavailable("Prompt exceeds the 20000-character limit.")
        limit = parameters.get("max_new_tokens", 128)
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 512:
            raise WorkerUnavailable("'max_new_tokens' must be an integer from 1 to 512.")
        do_sample = parameters.get("do_sample", False)
        if not isinstance(do_sample, bool):
            raise WorkerUnavailable("'do_sample' must be a boolean.")
        generation = {"max_new_tokens": limit, "do_sample": do_sample}
        if do_sample:
            temperature = parameters.get("temperature", 0.7)
            if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not 0 < temperature <= 2:
                raise WorkerUnavailable("'temperature' must be greater than 0 and at most 2.")
            generation["temperature"] = float(temperature)
        path = self._resolve_model_path(job.get("model"))
        torch, tokenizer, model = self._load(path)
        try:
            inputs = tokenizer(prompt, return_tensors="pt")
            device = next(model.parameters()).device
            inputs = {key: tensor.to(device) for key, tensor in inputs.items()}
            with torch.inference_mode():
                output = model.generate(**inputs, **generation)
            prompt_tokens = int(inputs["input_ids"].shape[-1])
            generated = output[0][prompt_tokens:]
            text = tokenizer.decode(generated, skip_special_tokens=True)
            return {"text": text, "model": path.name, "device": str(device),
                    "prompt_tokens": prompt_tokens, "generated_tokens": int(generated.shape[-1])}
        except Exception as exc:
            raise RuntimeError("Local model inference failed.") from exc
