

from __future__ import annotations

import threading

from onevoice.telemetry.stats import summarize

try:
    import pynvml
except ImportError:  # pragma: no cover - optional dependency
    pynvml = None  # type: ignore[assignment]

class NvmlGPUProfiler:

    def __init__(self, device_index: int = 0, sample_interval_s: float = 0.25) -> None:
        self._device_index = device_index
        self._interval = sample_interval_s
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._util_samples: list[float] = []
        self._mem_samples_mb: list[float] = []
        self._handle: object = None
        self._active = False
        self._total_mem_mb = 0.0

    @property
    def available(self) -> bool:
        return self._active

    def start_profiling(self) -> None:
        if pynvml is None:
            return
        try:
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self._device_index)
            mem = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            self._total_mem_mb = mem.total / (1024 * 1024)
        except Exception:  # pragma: no cover - driver-dependent
            self._handle = None
            return
        self._active = True
        self._stop.clear()
        self._util_samples.clear()
        self._mem_samples_mb.clear()
        self._thread = threading.Thread(
            target=self._loop, name="onevoice-gpuprofiler", daemon=True
        )
        self._thread.start()

    def stop_profiling(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._active and pynvml is not None:
            try:
                pynvml.nvmlShutdown()
            except Exception:  # pragma: no cover
                pass

    def _loop(self) -> None:
        assert pynvml is not None
        while not self._stop.wait(self._interval):
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
                self._util_samples.append(float(util.gpu))
                self._mem_samples_mb.append(mem.used / (1024 * 1024))
            except Exception:  # pragma: no cover
                break

    def get_metrics(self) -> dict[str, float]:
        if not self._active:
            return {}
        util = summarize(self._util_samples)
        mem = summarize(self._mem_samples_mb)
        headroom = 0.0
        if self._total_mem_mb > 0:
            headroom = 100.0 * (1.0 - mem["max"] / self._total_mem_mb)
        return {
            "gpu_util_percent_avg": util["avg"],
            "gpu_util_percent_max": util["max"],
            "gpu_mem_mb_avg": mem["avg"],
            "gpu_mem_mb_max": mem["max"],
            "gpu_mem_total_mb": self._total_mem_mb,
            "gpu_mem_headroom_percent": headroom,
        }
