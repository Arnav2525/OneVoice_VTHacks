from __future__ import annotations

import logging
import math
import threading
from collections import deque
from dataclasses import replace
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class LipGate:
    def __init__(self, hold_ms=500.0, attenuation_db=35.0):
        if hold_ms < 0 or not 0 <= attenuation_db <= 80:
            raise ValueError("Invalid lip gate settings")
        self.hold_ms = hold_ms
        self.floor = 10 ** (-attenuation_db / 20)
        self._lock = threading.Lock()
        self._history = deque(maxlen=600)
        self._key = None
        self._previous = None
        self._last_active = 0.0
        self.reset()

    def reset(self):
        self._gain = 1.0
        self._audio_key = None

    def observe(self, timestamp, key, aperture):
        with self._lock:
            if key != self._key:
                self._key = key
                self._history.clear()
                self._previous = None
                self._last_active = timestamp
            if self._history and timestamp <= self._history[-1][0]:
                return
            if self._history and timestamp - self._history[-1][0] > 250:
                self._previous = None
                self._last_active = timestamp
            state = None
            if aperture is not None and math.isfinite(aperture):
                movement = (abs(aperture - self._previous)
                            if self._previous is not None else 0.0)
                if aperture > 0.045 or movement > 0.012:
                    self._last_active = timestamp
                state = timestamp - self._last_active < self.hold_ms
                self._previous = aperture
            else:
                self._previous = None
                self._last_active = timestamp
            self._history.append((timestamp, state))

    def process(self, chunk):
        key = (chunk.metadata.get("target_track_id"),
               chunk.metadata.get("ui_selection_epoch"))
        with self._lock:
            history = list(self._history) if key == self._key else []
        if key != self._audio_key:
            self.reset()
            self._audio_key = key
        samples = np.asarray(chunk.data, dtype=np.float32)
        desired = np.ones(len(samples), dtype=np.float32)
        source = chunk.metadata.get("source_timestamp_ms")
        known = np.zeros(len(samples), dtype=bool)
        if history and isinstance(source, (int, float)) and math.isfinite(source):
            stamps = (source - float(chunk.metadata.get("denoise_delay_ms", 0))
                      + np.arange(len(samples)) * 1000 / chunk.sample_rate)
            times = np.array([t for t, _ in history])
            states = np.array([s is not False for _, s in history])
            measured = np.array([s is not None for _, s in history])
            indexes = np.searchsorted(times, stamps, side="right") - 1
            safe = np.clip(indexes, 0, len(times) - 1)
            known = (indexes >= 0) & (stamps - times[safe] <= 250) & measured[safe]
            desired[known & ~states[safe]] = self.floor
            for shift in range(1, 5):
                future = np.clip(safe + shift, 0, len(times) - 1)
                resume = ((times[future] <= stamps + 80)
                          & states[future] & measured[future])
                desired[resume] = 1.0
        gains = np.empty(len(samples), dtype=np.float32)
        step = 1000 / (chunk.sample_rate * 20)
        for i, target in enumerate(desired):
            self._gain += max(-step, min(step, float(target) - self._gain))
            gains[i] = self._gain
        return replace(chunk, data=(samples * gains).tolist(), metadata={
            **chunk.metadata,
            "lip_gate_gain_min": float(gains.min()) if len(gains) else 1.0,
            "lip_gate_known_fraction": float(known.mean()) if len(known) else 0.0,
        })


class LipSelector:
    def __init__(self, inner, state, gate, model_path):
        from mediapipe.tasks.python.core.base_options import BaseOptions
        from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions
        from mediapipe.tasks.python.vision.core import image

        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"Lip tracking model missing: {path}")
        self.inner, self.state, self.gate = inner, state, gate
        self._image = image
        self._model = FaceLandmarker.create_from_options(FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(path)),
            num_faces=1,
            min_face_detection_confidence=0.6,
            min_face_presence_confidence=0.6,
        ))
        self._timestamp = None
        self._warned = False

    def _measure(self, frame, track):
        if track is None or frame.data is None:
            return None
        data = np.asarray(frame.data)
        x, y, w, h = track.bounding_box
        top, left = max(0, int(y - h * 0.15)), max(0, int(x - w * 0.15))
        bottom = min(data.shape[0], int(y + h * 1.15))
        right = min(data.shape[1], int(x + w * 1.15))
        if bottom - top < 48 or right - left < 48:
            return None
        crop = np.ascontiguousarray(data[top:bottom, left:right, ::-1])
        result = self._model.detect(self._image.Image(
            image_format=self._image.ImageFormat.SRGB, data=crop
        ))
        if not result.face_landmarks:
            return None
        points = result.face_landmarks[0]

        def distance(a, b):
            return math.hypot((points[a].x - points[b].x) * crop.shape[1],
                              (points[a].y - points[b].y) * crop.shape[0])

        width = distance(61, 291)
        return distance(13, 14) / width if width >= 12 else None

    def select_target(self, frame, tracks):
        target = self.inner.select_target(frame, tracks)
        if frame.timestamp_ms != self._timestamp:
            self._timestamp = frame.timestamp_ms
            key = self.state.selection()
            track = target.selected_speaker
            try:
                aperture = self._measure(frame, track) if (
                    track is not None and track.track_id == key[0]
                ) else None
            except Exception:
                if not self._warned:
                    logger.exception("Lip tracking unavailable; preserving speech")
                    self._warned = True
                aperture = None
            self.gate.observe(frame.timestamp_ms, key, aperture)
        return target

    def close(self):
        self._model.close()
