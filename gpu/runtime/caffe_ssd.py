"""Opt-in Caffe SSD inference adapter for GQL.

This adapter expects a Caffe *deploy* prototxt plus trained caffemodel weights.
It supports the common DetectionOutput tensor [1, 1, N, 7] only; training
prototxts that expose raw loc/conf/prior-box tensors need a model-specific decoder.
"""
import os
from pathlib import Path
from typing import Any


_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class CaffeSSDWorker:
    """Run a trusted Caffe SSD model with 300x300 BGR mean-subtracted input."""

    def __init__(
        self,
        model_dir: str | None = None,
        input_dir: str | None = None,
    ) -> None:
        self.model_dir = Path(model_dir or os.getenv("MODEL_DIR", "/models")).expanduser().resolve()
        self.input_dir = Path(input_dir or os.getenv("GQL_INPUT_DIR", "/inputs")).expanduser().resolve()

    @staticmethod
    def _contained_file(root: Path, value: Any, suffixes: set[str]) -> Path:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("A non-empty relative file path is required")
        candidate = (root / value).resolve()
        if not candidate.is_relative_to(root):
            raise ValueError("File path must remain inside its configured directory")
        if candidate.suffix.lower() not in suffixes:
            raise ValueError(f"File extension must be one of: {', '.join(sorted(suffixes))}")
        if not candidate.is_file():
            raise ValueError(f"File not found in configured directory: {candidate.name}")
        return candidate

    def execute(self, job: dict[str, Any]) -> dict[str, Any]:
        if job.get("kind") != "inference":
            raise ValueError("Caffe SSD worker supports inference jobs only")
        parameters = job.get("parameters") or {}
        if not isinstance(parameters, dict):
            raise ValueError("Job parameters must be an object")

        prototxt = self._contained_file(self.model_dir, job.get("model"), {".prototxt"})
        weights = self._contained_file(
            self.model_dir, parameters.get("weights_path"), {".caffemodel"}
        )
        image_value = parameters.get("input_path")
        if not isinstance(image_value, str) or Path(image_value).suffix.lower() not in _IMAGE_SUFFIXES:
            raise ValueError("input_path must reference a supported image file")
        image_path = self._contained_file(self.input_dir, image_value, _IMAGE_SUFFIXES)

        threshold = parameters.get("confidence_threshold", 0.25)
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
            raise ValueError("confidence_threshold must be between 0 and 1")

        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("Caffe SSD inference requires OpenCV") from exc

        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError("Input file could not be decoded as an image")
        height, width = image.shape[:2]

        # Matches the supplied training configuration: 300x300 warp and BGR mean subtraction.
        blob = cv2.dnn.blobFromImage(
            image,
            scalefactor=1.0,
            size=(300, 300),
            mean=(104.0, 117.0, 123.0),
            swapRB=False,
            crop=False,
        )
        net = cv2.dnn.readNetFromCaffe(str(prototxt), str(weights))
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        net.setInput(blob)
        output = net.forward()
        detections = output.reshape(-1, output.shape[-1])
        if detections.shape[1] != 7:
            raise ValueError(
                "Unsupported Caffe SSD output: expected DetectionOutput rows with 7 values "
                "(image_id, class_id, confidence, xmin, ymin, xmax, ymax)"
            )

        results = []
        for row in detections:
            image_id, class_id, confidence, xmin, ymin, xmax, ymax = map(float, row)
            if image_id < 0 or confidence < float(threshold):
                continue
            x1 = max(0, min(width, int(round(xmin * width))))
            y1 = max(0, min(height, int(round(ymin * height))))
            x2 = max(x1, min(width, int(round(xmax * width))))
            y2 = max(y1, min(height, int(round(ymax * height))))
            results.append({
                "class_id": int(class_id),
                "confidence": round(confidence, 6),
                "box": [x1, y1, x2 - x1, y2 - y1],
            })

        return {
            "model": prototxt.name,
            "weights": weights.name,
            "backend": "cpu",
            "image": image_path.name,
            "image_size": [width, height],
            "detection_count": len(results),
            "detections": results,
            "output_format": "caffe_detection_output",
        }
