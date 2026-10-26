"""Opt-in YOLOv7 ONNX inference adapter for GQL.

The worker only reads model files under MODEL_DIR and input images under
GQL_INPUT_DIR. It never downloads models or executes user-supplied code.
"""
import os
from pathlib import Path
from typing import Any


class YOLOv7ONNXWorker:
    def __init__(self, model_dir: str | None = None, input_dir: str | None = None):
        self.model_dir = Path(model_dir or os.getenv("MODEL_DIR", "/models")).expanduser().resolve()
        self.input_dir = Path(input_dir or os.getenv("GQL_INPUT_DIR", "/inputs")).expanduser().resolve()

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
            raise ValueError("YOLOv7 ONNX worker supports inference jobs only")
        parameters = job.get("parameters") or {}
        if not isinstance(parameters, dict):
            raise ValueError("Job parameters must be an object")

        model_path = self._contained_file(self.model_dir, job.get("model"), suffix=".onnx")
        input_value = parameters.get("input_path")
        if not isinstance(input_value, str) or Path(input_value).suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
            raise ValueError("input_path must reference a JPG, JPEG, PNG, WEBP, or BMP image")
        image_path = self._contained_file(
            self.input_dir, input_value, suffix=Path(input_value).suffix.lower()
        )
        image_size = parameters.get("image_size", 640)
        if isinstance(image_size, bool) or not isinstance(image_size, int) or not 64 <= image_size <= 2048:
            raise ValueError("image_size must be an integer between 64 and 2048")
        confidence_threshold = parameters.get("confidence_threshold", 0.25)
        nms_threshold = parameters.get("nms_threshold", 0.45)
        for name, value in (("confidence_threshold", confidence_threshold), ("nms_threshold", nms_threshold)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1")

        try:
            import cv2
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("YOLOv7 inference requires OpenCV and NumPy") from exc

        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError("Input file could not be decoded as an image")
        original_height, original_width = image.shape[:2]
        blob = cv2.dnn.blobFromImage(
            image, scalefactor=1.0 / 255.0, size=(image_size, image_size),
            swapRB=True, crop=False,
        )
        net = cv2.dnn.readNetFromONNX(str(model_path))
        backend = str(parameters.get("backend", "auto")).lower()
        if backend not in {"auto", "cpu", "cuda"}:
            raise ValueError("backend must be one of: auto, cpu, cuda")
        actual_backend = "cpu"
        if backend == "cuda":
            if not hasattr(cv2.dnn, "DNN_BACKEND_CUDA") or not hasattr(cv2.dnn, "DNN_TARGET_CUDA"):
                raise RuntimeError("This OpenCV build does not expose CUDA DNN support")
            try:
                net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                actual_backend = "cuda"
            except Exception as exc:
                raise RuntimeError("OpenCV could not configure CUDA DNN") from exc
        else:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        net.setInput(blob)
        outputs = net.forward(net.getUnconnectedOutLayersNames())
        if isinstance(outputs, np.ndarray):
            outputs = [outputs]
        detections = []
        for output in outputs:
            rows = np.asarray(output)
            if rows.ndim == 3:
                rows = rows[0]
            if rows.ndim != 2 or rows.shape[1] < 6:
                continue
            boxes, scores, class_ids = [], [], []
            scale_x, scale_y = original_width / image_size, original_height / image_size
            for row in rows:
                objectness = float(row[4])
                class_scores = row[5:]
                if class_scores.size == 0:
                    continue
                class_id = int(np.argmax(class_scores))
                confidence = objectness * float(class_scores[class_id])
                if confidence < confidence_threshold:
                    continue
                center_x, center_y, width, height = map(float, row[:4])
                left = int((center_x - width / 2) * scale_x)
                top = int((center_y - height / 2) * scale_y)
                box_width = int(width * scale_x)
                box_height = int(height * scale_y)
                boxes.append([left, top, box_width, box_height])
                scores.append(confidence)
                class_ids.append(class_id)
            keep = cv2.dnn.NMSBoxes(boxes, scores, float(confidence_threshold), float(nms_threshold))
            for index in np.asarray(keep).reshape(-1).tolist() if len(keep) else []:
                x, y, width, height = boxes[int(index)]
                detections.append({
                    "class_id": class_ids[int(index)],
                    "confidence": round(scores[int(index)], 6),
                    "box": [max(0, x), max(0, y), max(0, width), max(0, height)],
                })

        return {
            "model": model_path.name,
            "backend": actual_backend,
            "image": image_path.name,
            "image_size": [original_width, original_height],
            "detection_count": len(detections),
            "detections": detections,
        }
