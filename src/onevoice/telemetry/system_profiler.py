

from __future__ import annotations

import threading
import time
from collections.abc import Callable

from onevoice.telemetry.stats import summarize

try:
    import psutil
except ImportError:  # pragma: no cover - optional dependency
    psutil = None  # type: ignore[assignment]

class SystemProfiler:

    def __init__(self, sample_interval_s: float = 0.25) -> None:
        self._interval = sample_interval_s
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._cpu_samples: list[float] = []
        self._ram_samples_mb: list[float] = []
        self._proc = psutil.Process() if psutil is not None else None

    @property
    def available(self) -> bool:
        return psutil is not None

    def start_profiling(self) -> None:
        if psutil is None:
            return
        self._stop.clear()
        self._cpu_samples.clear()
        self._ram_samples_mb.clear()
        assert self._proc is not None
        self._proc.cpu_percent(None)
        self._thread = threading.Thread(
            target=self._loop, name="onevoice-sysprofiler", daemon=True
        )
        self._thread.start()

    def stop_profiling(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _loop(self) -> None:
        assert self._proc is not None and psutil is not None
        while not self._stop.wait(self._interval):
            try:
                self._cpu_samples.append(self._proc.cpu_percent(None))
                self._ram_samples_mb.append(
                    self._proc.memory_info().rss / (1024 * 1024)
                )
            except Exception:  # pragma: no cover - process may vanish
                break

    def get_metrics(self) -> dict[str, float]:
        if psutil is None:
            return {}
        metrics: dict[str, float] = {}
        cpu = summarize(self._cpu_samples)
        ram = summarize(self._ram_samples_mb)
        metrics["cpu_percent_avg"] = cpu["avg"]
        metrics["cpu_percent_max"] = cpu["max"]
        metrics["ram_mb_avg"] = ram["avg"]
        metrics["ram_mb_max"] = ram["max"]
        try:
            metrics["system_cpu_percent"] = psutil.cpu_percent(interval=None)
        except Exception:  # pragma: no cover
            pass
        return metrics

def measure_import_time(import_callable: Callable[[], object]) -> float:

    start = time.perf_counter()
    import_callable()
    return (time.perf_counter() - start) * 1000.0
