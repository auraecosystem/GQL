"""Read-only diagnostics for the GQL runtime and mounted workload directories.

This module deliberately does not scan the LAN, contact third-party services,
install drivers, or mutate host configuration.
"""
import os
from pathlib import Path
import shutil
import subprocess


def _command_output(command: str, *args: str) -> str | None:
    """Run a known local diagnostic executable with a short timeout."""
    executable = shutil.which(command)
    if not executable:
        return None
    try:
        result = subprocess.run(
            [executable, *args],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return (result.stdout or result.stderr).strip()[:1000] or None


def _directory_check(path_value: str) -> dict[str, bool]:
    path = Path(path_value)
    try:
        return {
            "exists": path.is_dir(),
            "readable": path.is_dir() and os.access(path, os.R_OK | os.X_OK),
        }
    except OSError:
        return {"exists": False, "readable": False}


def collect_diagnostics() -> dict:
    """Return best-effort, read-only GPU runtime and mount diagnostics."""
    nvidia_smi = _command_output("nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader")
    nvcc = _command_output("nvcc", "--version")
    rocminfo = _command_output("rocminfo")
    model_dir = os.getenv("MODEL_DIR", "/models")
    input_dir = os.getenv("GQL_INPUT_DIR", "/inputs")

    nvidia_devices = sorted(str(path) for path in Path("/dev").glob("nvidia*"))
    checks = {
        "nvidia_smi_available": nvidia_smi is not None,
        "cuda_toolkit_available": nvcc is not None,
        "nvidia_device_nodes_present": bool(nvidia_devices),
        "rocm_runtime_available": rocminfo is not None,
        "rocm_device_nodes_present": Path("/dev/kfd").exists() or Path("/dev/dri").exists(),
        "model_directory": _directory_check(model_dir),
        "input_directory": _directory_check(input_dir),
    }

    if nvidia_smi is not None:
        backend = "cuda"
    elif rocminfo is not None:
        backend = "rocm"
    else:
        backend = "cpu_or_gpu_not_exposed"

    return {
        "status": "ok" if checks["model_directory"]["exists"] and checks["input_directory"]["exists"] else "degraded",
        "backend_hint": backend,
        "checks": checks,
        "nvidia_smi": nvidia_smi,
        "cuda_toolkit": nvcc,
        "rocminfo": rocminfo,
        "nvidia_device_nodes": nvidia_devices,
        "directories": {"model_dir": model_dir, "input_dir": input_dir},
        "notes": [
            "GPU detection is best-effort and does not prove model-level compatibility.",
            "A missing GPU runtime may be expected for CPU-only deployments.",
            "Diagnostics are read-only; no network scanning or external vendor lookups are performed.",
        ],
    }
