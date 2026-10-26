import tempfile
import unittest
from pathlib import Path
from runtime.local_worker import LocalTransformersWorker
from runtime.workers import WorkerUnavailable

class LocalTransformersWorkerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.models = Path(self.directory.name)
        (self.models / "tiny-model").mkdir()
        (self.models / "tiny-model" / "config.json").write_text("{}", encoding="utf-8")
        self.worker = LocalTransformersWorker(str(self.models))
    def tearDown(self):
        self.directory.cleanup()
    def test_rejects_model_path_escape(self):
        with self.assertRaises(WorkerUnavailable): self.worker._resolve_model_path("../outside")
    def test_rejects_missing_model(self):
        with self.assertRaises(WorkerUnavailable): self.worker._resolve_model_path("not-installed")
    def test_rejects_non_inference_job(self):
        with self.assertRaises(WorkerUnavailable):
            self.worker.execute({"kind":"training","model":"tiny-model","parameters":{}})
    def test_requires_prompt(self):
        with self.assertRaises(WorkerUnavailable):
            self.worker.execute({"kind":"inference","model":"tiny-model","parameters":{}})
    def test_rejects_invalid_token_limit(self):
        with self.assertRaises(WorkerUnavailable):
            self.worker.execute({"kind":"inference","model":"tiny-model","parameters":{"prompt":"hello","max_new_tokens":100000}})
if __name__ == "__main__": unittest.main()
