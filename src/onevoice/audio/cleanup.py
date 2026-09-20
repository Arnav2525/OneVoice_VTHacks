

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

import numpy as np

from onevoice.core.models.audio_chunk import AudioChunk

class RumbleFilter:

    def __init__(self, cutoff_hz: float = 80.0) -> None:
        if not math.isfinite(cutoff_hz) or cutoff_hz <= 0:
            raise ValueError("cleanup cutoff_hz must be finite and positive")
        self.cutoff_hz = cutoff_hz
        self.reset()

    def reset(self) -> None:
        self._key: Any = None
        self._previous_input: Any = None
        self._previous_output: Any = None

    def process(self, chunk: AudioChunk) -> AudioChunk:
        rate, channels = chunk.sample_rate, chunk.channels
        if rate <= 0 or channels <= 0 or self.cutoff_hz >= rate / 2:
            raise ValueError("cleanup requires valid channels and cutoff below Nyquist")
        samples = np.asarray(chunk.data, dtype=np.float32).reshape(-1)
        if samples.size % channels or not np.isfinite(samples).all():
            raise ValueError("cleanup received invalid audio")
        key = (
            rate,
            channels,
            chunk.metadata.get("target_track_id"),
            chunk.metadata.get("ui_selection_epoch"),
        )
        if key != self._key:
            self.reset()
            self._key = key
            self._previous_input = np.zeros(channels, dtype=np.float64)
            self._previous_output = np.zeros(channels, dtype=np.float64)
        output = np.empty_like(samples).reshape(-1, channels)

        alpha = 1.0 / (1.0 + 2.0 * math.pi * self.cutoff_hz / rate)
        for c in range(channels):
            previous_input = float(self._previous_input[c])
            previous_output = float(self._previous_output[c])
            for i, value in enumerate(samples[c::channels]):
                current = float(value)
                previous_output = alpha * (previous_output + current - previous_input)
                output[i, c] = previous_output
                previous_input = current
            self._previous_input[c] = previous_input
            self._previous_output[c] = previous_output
        return replace(
            chunk,
            data=output.reshape(-1).tolist(),
            metadata={**chunk.metadata, "rumble_filter_hz": self.cutoff_hz},
        )
