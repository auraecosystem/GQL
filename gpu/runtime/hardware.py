"""Best-effort hardware discovery without requiring a GPU framework."""
import os
import platform
import shutil
import subprocess


def _run(command):
    executable = shutil.which(command[0])
    if not executable:
        return None
    try:
        result = subprocess.run([executable, *command[1:]], capture_output=True, text=True, timeout=2, check=False)
        return result.stdout.strip()[:2000] if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def detect_hardware():
    nvidia = _run(["nvidia-smi", "-L"])
    rocm = _run(["rocminfo"])
    torch_info = {"available": False, "cuda_available": False, "device_count": 0}
    try:
        import torch
        available = bool(torch.cuda.is_available())
        torch_info = {
            "available": True,
            "cuda_available": available,
            "device_count": int(torch.cuda.device_count()) if available else 0,
            "version": str(torch.__version__),
        }
    except Exception:
        pass
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "backend_hint": "cuda" if nvidia else "rocm" if rocm else "cpu_or_gpu_not_exposed",
        "nvidia_smi": nvidia,
        "rocminfo_detected": rocm is not None,
        "torch": torch_info,
        "model_dir": os.getenv("MODEL_DIR", "/models"),
    }
