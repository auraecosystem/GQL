"""Opt-in OpenCV DNN Caffe SSD inference worker.

Expected output follows the common SSD DetectionOutput layout [N, 1, K, 7]
with rows [image_id, class_id, confidence, x1, y1, x2, y2]. Model artifacts
are operator-provided; this adapter never downloads models or executes user code.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any


class CaffeSSDWorker:
    def __init__(self, model_dir: str | None = None, input_dir: str | None = None,
                 config_path: str | None = None) -> None:
        self.model_dir = Path(model_dir or os.getenv("MODEL_DIR", "/models")).expanduser().resolve()
        self.input_dir = Path(input_dir or os.getenv("GQL_INPUT_DIR", "/inputs")).expanduser().resolve()
        configured = config_path or os.getenv("GQL_CAFFE_SSD_CONFIG")
        self.config_path = Path(configured).expanduser().resolve() if configured else (self.model_dir / "deploy.prototxt").resolve()

    @staticmethod
    def _contained_file(root: Path, value: Any, *, suffix: str) -> Path:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("A non-empty relative file path is required")
        candidate = (root / value).resolve()
        if not candidate.is_relative_to(root):
            raise ValueError("File path must remain inside its configured directory")
        if candidate.suffix.lower() != suffix:
            raise ValueError(f"File must use the {suffix} extension")
        if not candidate.is_file():
            raise ValueError(f"File not found in configured directory: {candidate.name}")
        return candidate

    def execute(self, job: dict[str, Any]) -> dict[str, Any]:
        if job.get("kind") != "inference":
            raise ValueError("Caffe SSD worker supports inference jobs only")
        parameters = job.get("parameters") or {}
        if not isinstance(parameters, dict):
            raise ValueError("Job parameters must be an object")
        model_path = self._contained_file(self.model_dir, job.get("model"), suffix=".caffemodel")
        config_path = self.config_path.resolve()
        if not config_path.is_relative_to(self.model_dir) or not config_path.is_file():
            raise ValueError("Caffe SSD deploy prototxt must exist inside MODEL_DIR or be configured inside MODEL_DIR")
        input_value = parameters.get("input_path")
        if not isinstance(input_value, str) or Path(input_value).suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
            raise ValueError("input_path must reference a JPG, JPEG, PNG, WEBP, or BMP image")
        image_path = self._contained_file(self.input_dir, input_value, suffix=Path(input_value).suffix.lower())
        threshold = parameters.get("confidence_threshold", 0.5)
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
            raise ValueError("confidence_threshold must be between 0 and 1")
        width = parameters.get("input_width", 300)
        height = parameters.get("input_height", 300)
        for name, value in (("input_width", width), ("input_height", height)):
            if isinstance(value, bool) or not isinstance(value, int) or not 32 <= value <= 2048:
                raise ValueError(f"{name} must be an integer between 32 and 2048")
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("Caffe SSD inference requires OpenCV") from exc
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError("Input file could not be decoded as an image")
        net = cv2.dnn.readNetFromCaffe(str(config_path), str(model_path))
        backend = str(parameters.get("backend", "cpu")).lower()
        if backend not in {"auto", "cpu"}:
            raise ValueError("Caffe SSD currently supports backend 'cpu' or 'auto'")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        blob = cv2.dnn.blobFromImage(
            image, scalefactor=1.0, size=(width, height),
            mean=(104.0, 177.0, 123.0), swapRB=False, crop=False,
        )
        net.setInput(blob)
        output = net.forward()
        detections = []
        for row in output.reshape(-1, output.shape[-1]):
            if len(row) < 7:
                continue
            confidence = float(row[2])
            if confidence < float(threshold):
                continue
            x1, y1, x2, y2 = map(float, row[3:7])
            left = max(0, min(image.shape[1], int(x1 * image.shape[1])))
            top = max(0, min(image.shape[0], int(y1 * image.shape[0])))
            right = max(left, min(image.shape[1], int(x2 * image.shape[1])))
            bottom = max(top, min(image.shape[0], int(y2 * image.shape[0])))
            detections.append({
                "class_id": int(row[1]),
                "confidence": round(confidence, 6),
                "box": [left, top, right - left, bottom - top],
            })
        return {
            "model": model_path.name, "backend": "cpu", "image": image_path.name,
            "image_size": [int(image.shape[1]), int(image.shape[0])],
            "detections": detections, "count": len(detections),
        }
