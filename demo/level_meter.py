

from __future__ import annotations

import array
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np

from onevoice.core.models.audio_chunk import AudioChunk

def _chunk_to_float32(chunk: AudioChunk) -> np.ndarray:

    data = chunk.data
    if isinstance(data, array.array):
        return np.frombuffer(data, dtype=np.float32)
    if hasattr(data, "astype"):
        return np.asarray(data, dtype=np.float32).reshape(-1)
    return np.array(data, dtype=np.float32)

class LevelMeterSink:

    def __init__(
        self,
        sample_rate: int = 16_000,
        rms_smoothing_alpha: float = 0.3,
        peak_decay_s: float = 1.2,
    ) -> None:
        self._sample_rate = sample_rate
        self._rms_alpha = rms_smoothing_alpha
        self._peak_decay_s = peak_decay_s
        self._lock = threading.Lock()
        self._rms = 0.0
        self._peak = 0.0
        self._last_update = time.monotonic()

    def start(self) -> None:
        with self._lock:
            self._rms = 0.0
            self._peak = 0.0
            self._last_update = time.monotonic()

    def stop(self) -> None:
        pass

    def write(self, chunk: AudioChunk) -> None:
        samples = _chunk_to_float32(chunk)
        if len(samples) == 0:
            return
        now = time.monotonic()
        chunk_rms = float(np.sqrt(np.mean(np.square(samples))))
        chunk_peak = float(np.max(np.abs(samples)))
        with self._lock:
            elapsed = max(0.0, now - self._last_update)
            self._last_update = now
            self._rms = (
                self._rms_alpha * chunk_rms + (1.0 - self._rms_alpha) * self._rms
            )
            decay = (
                elapsed / self._peak_decay_s if self._peak_decay_s > 0 else self._peak
            )
            self._peak = max(chunk_peak, self._peak - decay, 0.0)

    def snapshot(self) -> tuple[float, float]:

        with self._lock:
            return min(1.0, max(0.0, self._rms)), min(1.0, max(0.0, self._peak))
