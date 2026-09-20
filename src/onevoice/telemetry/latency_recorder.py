

from __future__ import annotations

import threading
from collections import defaultdict

class StageLatencyRecorder:

    def __init__(self, window_size: int = 128) -> None:
        self._window_size = window_size
        self._lock = threading.Lock()
        self._starts: dict[str, float] = {}
        self._history: defaultdict[str, list[float]] = defaultdict(list)
        self._last: dict[str, float] = {}

    def record_start(self, metric_name: str, timestamp_ms: float) -> None:
        with self._lock:
            self._starts[metric_name] = timestamp_ms

    def record_end(self, metric_name: str, timestamp_ms: float) -> None:
        with self._lock:
            start = self._starts.pop(metric_name, timestamp_ms)
            duration = max(0.0, timestamp_ms - start)
            history = self._history[metric_name]
            history.append(duration)
            if len(history) > self._window_size:
                del history[: len(history) - self._window_size]
            self._last[metric_name] = duration

    def get_latencies(self) -> dict[str, float]:
        with self._lock:
            result: dict[str, float] = {}
            for name, values in self._history.items():
                if values:
                    result[name] = sum(values) / len(values)
            result.update({f"{k}_last": v for k, v in self._last.items()})
            return result
