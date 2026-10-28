import os
import tempfile
import unittest
from unittest.mock import patch

from runtime.diagnostics import collect_diagnostics


class RuntimeDiagnosticsTests(unittest.TestCase):
    def test_reports_directory_readiness_without_gpu_hardware(self):
        with tempfile.TemporaryDirectory() as root:
            models = os.path.join(root, "models")
            inputs = os.path.join(root, "inputs")
            os.mkdir(models)
            os.mkdir(inputs)
            with patch.dict(os.environ, {"MODEL_DIR": models, "GQL_INPUT_DIR": inputs}), patch(
                "runtime.diagnostics._command_output", return_value=None
            ):
                result = collect_diagnostics()

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["backend_hint"], "cpu_or_gpu_not_exposed")
        self.assertTrue(result["checks"]["model_directory"]["readable"])
        self.assertTrue(result["checks"]["input_directory"]["readable"])
        self.assertFalse(result["checks"]["nvidia_smi_available"])
        self.assertFalse(result["checks"]["cuda_toolkit_available"])

    def test_missing_mounts_are_reported_as_degraded(self):
        with tempfile.TemporaryDirectory() as root:
            with patch.dict(
                os.environ,
                {"MODEL_DIR": os.path.join(root, "missing-models"),
                 "GQL_INPUT_DIR": os.path.join(root, "missing-inputs")},
            ), patch("runtime.diagnostics._command_output", return_value=None):
                result = collect_diagnostics()

        self.assertEqual(result["status"], "degraded")
        self.assertFalse(result["checks"]["model_directory"]["exists"])
        self.assertFalse(result["checks"]["input_directory"]["exists"])

    def test_gpu_runtime_hint_uses_local_probe_results(self):
        def probe(command, *args):
            return "GPU 0, 550.54, 24576 MiB" if command == "nvidia-smi" else None

        with tempfile.TemporaryDirectory() as root:
            models = os.path.join(root, "models")
            inputs = os.path.join(root, "inputs")
            os.mkdir(models)
            os.mkdir(inputs)
            with patch.dict(os.environ, {"MODEL_DIR": models, "GQL_INPUT_DIR": inputs}), patch(
                "runtime.diagnostics._command_output", side_effect=probe
            ):
                result = collect_diagnostics()

        self.assertEqual(result["backend_hint"], "cuda")
        self.assertTrue(result["checks"]["nvidia_smi_available"])
        self.assertIn("GPU 0", result["nvidia_smi"])
        self.assertTrue(any("no network scanning" in note for note in result["notes"]))


if __name__ == "__main__":
    unittest.main()
