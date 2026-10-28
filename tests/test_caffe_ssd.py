import tempfile
import unittest
from pathlib import Path

from runtime.caffe_ssd import CaffeSSDWorker


class CaffeSSDWorkerValidationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.models = root / "models"
        self.inputs = root / "inputs"
        self.models.mkdir()
        self.inputs.mkdir()
        (self.models / "deploy.prototxt").write_text("deploy", encoding="utf-8")
        (self.models / "weights.caffemodel").write_bytes(b"test")
        (self.inputs / "frame.jpg").write_bytes(b"not-a-real-image")
        self.worker = CaffeSSDWorker(str(self.models), str(self.inputs))

    def tearDown(self):
        self.temporary.cleanup()

    def test_rejects_model_path_escape(self):
        with self.assertRaisesRegex(ValueError, "inside its configured directory"):
            self.worker.execute({
                "kind": "inference", "model": "../outside.caffemodel",
                "parameters": {"input_path": "frame.jpg"},
            })

    def test_rejects_wrong_model_extension(self):
        with self.assertRaisesRegex(ValueError, "extension"):
            self.worker.execute({
                "kind": "inference", "model": "deploy.prototxt",
                "parameters": {"input_path": "frame.jpg"},
            })

    def test_rejects_input_path_escape(self):
        with self.assertRaisesRegex(ValueError, "inside its configured directory"):
            self.worker.execute({
                "kind": "inference", "model": "weights.caffemodel",
                "parameters": {"input_path": "../outside.jpg"},
            })

    def test_rejects_non_inference_job(self):
        with self.assertRaisesRegex(ValueError, "inference jobs only"):
            self.worker.execute({
                "kind": "training", "model": "weights.caffemodel", "parameters": {},
            })

    def test_rejects_invalid_confidence_threshold(self):
        with self.assertRaisesRegex(ValueError, "confidence_threshold"):
            self.worker.execute({
                "kind": "inference", "model": "weights.caffemodel",
                "parameters": {"input_path": "frame.jpg", "confidence_threshold": 2},
            })

    def test_requires_deploy_prototxt(self):
        (self.models / "deploy.prototxt").unlink()
        with self.assertRaisesRegex(ValueError, "deploy prototxt"):
            self.worker.execute({
                "kind": "inference", "model": "weights.caffemodel",
                "parameters": {"input_path": "frame.jpg"},
            })


if __name__ == "__main__":
    unittest.main()
