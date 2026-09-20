

from __future__ import annotations

import platform
import sys
from typing import Any

try:
    import psutil
except ImportError:  # pragma: no cover - optional dependency
    psutil = None  # type: ignore[assignment]

try:
    import pynvml
except ImportError:  # pragma: no cover - optional dependency
    pynvml = None  # type: ignore[assignment]

def _gpu_info() -> list[dict[str, Any]]:
    if pynvml is None:
        return []
    gpus: list[dict[str, Any]] = []
    try:
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        for i in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpus.append(
                {
                    "index": i,
                    "name": name.decode() if isinstance(name, bytes) else name,
                    "total_memory_mb": round(mem.total / (1024 * 1024), 1),
                }
            )
        pynvml.nvmlShutdown()
    except Exception:  # pragma: no cover - driver-dependent
        return []
    return gpus

def _torch_info() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"available": False}
    return {
        "available": True,
        "version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": getattr(torch.version, "cuda", None),
    }

def collect_hardware_info() -> dict[str, Any]:

    info: dict[str, Any] = {
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": sys.version.split()[0],
    }
    if psutil is not None:
        info["cpu_count_logical"] = psutil.cpu_count(logical=True)
        info["cpu_count_physical"] = psutil.cpu_count(logical=False)
        vm = psutil.virtual_memory()
        info["total_ram_mb"] = round(vm.total / (1024 * 1024), 1)
    else:
        info["cpu_count_logical"] = None
        info["cpu_count_physical"] = None
        info["total_ram_mb"] = None
    info["gpus"] = _gpu_info()
    info["torch"] = _torch_info()
    return info
