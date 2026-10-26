import tempfile
import unittest
from pathlib import Path

from runtime.caffe_ssd import CaffeSSDWorker


class CaffeSSDWorkerValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.models = root / "models"
        self.inputs = root / "inputs"
        self.models.mkdir()
        self.inputs.mkdir()
        (self.models / "deploy.prototxt").write_text("placeholder", encoding="utf-8")
        (self.models / "weights.caffemodel").write_bytes(b"placeholder")
        (self.inputs / "frame.jpg").write_bytes(b"placeholder")
        self.worker = CaffeSSDWorker(str(self.models), str(self.inputs))

    def tearDown(self):
        self.temp.cleanup()

    def _job(self, **parameters):
        return {
            "kind": "inference",
            "model": "deploy.prototxt",
            "parameters": {"weights_path": "weights.caffemodel", "input_path": "frame.jpg", **parameters},
        }

    def test_rejects_model_path_traversal(self):
        job = self._job()
        job["model"] = "../outside.prototxt"
        with self.assertRaisesRegex(ValueError, "inside its configured directory"):
            self.worker.execute(job)

    def test_rejects_weights_path_traversal(self):
        with self.assertRaisesRegex(ValueError, "inside its configured directory"):
            self.worker.execute(self._job(weights_path="../outside.caffemodel"))

    def test_rejects_input_path_traversal(self):
        with self.assertRaisesRegex(ValueError, "inside its configured directory"):
            self.worker.execute(self._job(input_path="../secret.jpg"))

    def test_rejects_wrong_model_extension(self):
        job = self._job()
        job["model"] = "weights.caffemodel"
        with self.assertRaisesRegex(ValueError, "File extension"):
            self.worker.execute(job)

    def test_rejects_non_inference_job(self):
        job = self._job()
        job["kind"] = "training"
        with self.assertRaisesRegex(ValueError, "inference jobs only"):
            self.worker.execute(job)

    def test_rejects_invalid_confidence_threshold_before_loading_runtime(self):
        with self.assertRaisesRegex(ValueError, "confidence_threshold"):
            self.worker.execute(self._job(confidence_threshold=1.5))


if __name__ == "__main__":
    unittest.main()
