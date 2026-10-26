import unittest
from runtime.hardware import detect_hardware


class HardwareTests(unittest.TestCase):
    def test_detection_has_stable_contract(self):
        info = detect_hardware()
        self.assertIn("backend_hint", info)
        self.assertIn(info["backend_hint"], {"cuda", "rocm", "cpu_or_gpu_not_exposed"})
        self.assertIn("torch", info)
        self.assertIn("cuda_available", info["torch"])


if __name__ == "__main__":
    unittest.main()
