import tempfile
import unittest
from pathlib import Path

from runtime.yolov7 import YOLOv7ONNXWorker


class YOLOv7WorkerValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.models = root / "models"
        self.inputs = root / "inputs"
        self.models.mkdir()
        self.inputs.mkdir()
        (self.models / "detector.onnx").write_bytes(b"test-model-placeholder")
        (self.inputs / "frame.jpg").write_bytes(b"test-image-placeholder")
        self.worker = YOLOv7ONNXWorker(str(self.models), str(self.inputs))

    def tearDown(self):
        self.temp.cleanup()

    def test_rejects_model_path_traversal(self):
        with self.assertRaisesRegex(ValueError, "inside its configured directory"):
            self.worker.execute({
                "kind": "inference",
                "model": "../outside.onnx",
                "parameters": {"input_path": "frame.jpg"},
            })

    def test_rejects_wrong_model_extension(self):
        with self.assertRaisesRegex(ValueError, "onnx extension"):
            self.worker.execute({
                "kind": "inference",
                "model": "frame.jpg",
                "parameters": {"input_path": "frame.jpg"},
            })

    def test_rejects_input_path_traversal(self):
        with self.assertRaisesRegex(ValueError, "inside its configured directory"):
            self.worker.execute({
                "kind": "inference",
                "model": "detector.onnx",
                "parameters": {"input_path": "../secret.jpg"},
            })

    def test_rejects_non_inference_job(self):
        with self.assertRaisesRegex(ValueError, "inference jobs only"):
            self.worker.execute({"kind": "training", "model": "detector.onnx", "parameters": {}})

    def test_rejects_invalid_image_size_before_loading_runtime(self):
        with self.assertRaisesRegex(ValueError, "image_size"):
            self.worker.execute({
                "kind": "inference",
                "model": "detector.onnx",
                "parameters": {"input_path": "frame.jpg", "image_size": 12},
            })


if __name__ == "__main__":
    unittest.main()
