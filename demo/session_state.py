

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

@dataclass(frozen=True)
class SessionView:
    phase: str
    title: str
    detail: str
    mode: str
    active: bool
    selected_id: str | None
    person: str | None
    input_level: float
    output_level: float
    elapsed_s: float
    face_count: int

def level(data: Any) -> float:
    samples = np.asarray(data, dtype=np.float32)
    if not samples.size or not np.isfinite(samples).all():
        return 0.0
    return min(1.0, float(np.sqrt(np.mean(np.square(samples)))))

class SessionState:

    def __init__(
        self,
        mode: str = "preview",
        clock: Callable[[], float] = time.monotonic,
        stale_s: float = 2.0,
    ) -> None:
        self._lock = threading.RLock()
        self.clock = clock
        self.mode = mode
        self.stale_s = stale_s
        self._phase = "ready"
        self._detail = ""
        self._session_start = 0.0
        self._reset()

    def _reset(self) -> None:
        self._audio_at: float | None = None
        self._video_at: float | None = None
        self._tracks_at: float | None = None
        self._playback_at: float | None = None
        self._audio_stamp: float | None = None
        self._video_stamp: float | None = None
        self._input_level = 0.0
        self._output_level = 0.0
        self._output_ready = False
        self._backend_unavailable = False
        self._source_floor_ms = 0.0
        self._requested: str | None = None
        self._epoch = getattr(self, "_epoch", 0) + 1
        self._lost = False
        self._visible: set[str] = set()
        self._names: dict[str, str] = {}

    def begin_start(self) -> None:
        with self._lock:
            self._reset()
            self._phase, self._detail = "starting", "Connecting devices..."
            self._session_start = self.clock()
            self._source_floor_ms = self._session_start * 1000.0

    def starting_detail(self, detail: str) -> None:
        with self._lock:
            if self._phase == "starting":
                self._detail = detail

    def mark_running(self) -> None:
        with self._lock:
            if self._phase == "starting":
                self._phase = "running"
                self._session_start = self.clock()

    def begin_stop(self) -> None:
        with self._lock:
            if self._phase in ("starting", "running", "error"):
                self._phase = "stopping"
                self._epoch += 1
                self._output_ready = False

    def mark_stopped(self, error: str | None = None) -> None:
        with self._lock:
            self._reset()
            self._phase = "error" if error else "stopped"
            self._detail = error or "Camera and microphone released."

    def observe_input(self, chunk: Any) -> None:
        with self._lock:
            if self._phase not in ("starting", "running"):
                return
            if not self._valid_stamp(chunk.timestamp_ms, self._audio_stamp):
                return
            try:
                samples = np.asarray(chunk.data, dtype=np.float32)
            except (TypeError, ValueError):
                return
            if not samples.size or not np.isfinite(samples).all():
                return
            self._audio_stamp = float(chunk.timestamp_ms)
            self._audio_at = self._audio_stamp / 1000.0
            self._input_level = level(samples)

    def observe_frame(self, frame: Any) -> None:
        with self._lock:
            if self._phase not in ("starting", "running"):
                return
            if self.mode == "live" and frame.data is None:
                return
            if self._valid_stamp(frame.timestamp_ms, self._video_stamp):
                self._video_stamp = float(frame.timestamp_ms)
                self._video_at = self._video_stamp / 1000.0

    def observe_tracks(self, tracks: Sequence[Any]) -> None:
        with self._lock:
            if self._phase not in ("starting", "running"):
                return
            self._tracks_at = self.clock()
            self._visible = {
                t.track_id for t in tracks if t.metadata.get("visible", True)
            }
            for track in tracks:
                if track.track_id not in self._names:
                    number = track.metadata.get("person_number", len(self._names) + 1)
                    self._names[track.track_id] = f"Person {number}"
            if self._requested and self._requested not in self._visible:
                self._lost = True
                self._output_ready = False

    def person_name(self, track_id: str) -> str:
        with self._lock:
            return self._names.get(track_id, "Person")

    def select(self, track_id: str | None) -> bool:
        with self._lock:
            if self._phase != "running":
                return False
            if track_id is not None and track_id not in self._visible:
                return False
            self._requested = track_id
            self._epoch += 1
            self._lost = False
            self._output_ready = False
            self._output_level = 0.0
            return True

    def selection(self) -> tuple[str | None, int]:
        with self._lock:
            selected = self._requested if not self._lost else None
            return selected, self._epoch

    def output_allowed(self, chunk: Any) -> bool:
        with self._lock:
            try:
                samples = np.asarray(chunk.data, dtype=np.float32)
            except (TypeError, ValueError):
                return False
            if not samples.size or not np.isfinite(samples).all():
                return False
            meta = chunk.metadata
            fresh = all(
                self._fresh(t)
                for t in (
                    self._audio_at,
                    self._video_at,
                    self._tracks_at,
                )
            )
            return bool(
                self._phase == "running"
                and self.mode == "live"
                and fresh
                and self._requested
                and not self._lost
                and self._requested in self._visible
                and meta.get("ui_selection_epoch") == self._epoch
                and meta.get("target_track_id") == self._requested
                and meta.get("conditioning") == "visual"
                and meta.get("output_ready") is True
                and not meta.get("fallback")
                and not meta.get("passthrough")
                and self._valid_stamp(chunk.timestamp_ms)
            )

    def observe_output(self, chunk: Any, backend: dict[str, Any]) -> None:
        with self._lock:
            if self._phase != "running":
                return
            if not self._valid_stamp(chunk.timestamp_ms):
                self._output_ready = False
                self._output_level = 0.0
                return
            self._playback_at = self.clock()
            self._backend_unavailable = bool(
                self.mode == "live"
                and (
                    backend.get("is_real_separation") is not True
                    or backend.get("fallback_active")
                    or chunk.metadata.get("fallback")
                    or chunk.metadata.get("passthrough")
                )
            )
            self._output_ready = bool(
                self.output_allowed(chunk)
                and backend.get("is_real_separation") is True
                and not backend.get("fallback_active")
            )
            self._output_level = level(chunk.data) if self._output_ready else 0.0

    def _valid_stamp(self, timestamp_ms: Any, previous: float | None = None) -> bool:

        try:
            stamp = float(timestamp_ms)
        except (TypeError, ValueError):
            return False
        now_ms = self.clock() * 1000.0
        return (
            math.isfinite(stamp)
            and stamp >= self._source_floor_ms
            and (previous is None or stamp > previous)
            and 0.0 <= now_ms - stamp <= self.stale_s * 1000.0
        )

    def _fresh(self, at: float | None) -> bool:
        return at is not None and self.clock() - at <= self.stale_s

    def snapshot(self) -> SessionView:
        with self._lock:
            phase = self._phase
            detail = self._detail
            person = self._names.get(self._requested or "")
            elapsed = max(0.0, self.clock() - self._session_start)
            title = {
                "ready": "Ready to listen",
                "starting": "Connecting devices...",
                "stopping": "Stopping session...",
                "stopped": "Session stopped",
                "error": "Session could not continue",
            }.get(phase, "Listening")
            if phase == "ready":
                detail = "Press Start listening when you are ready."
            if phase == "starting":
                title = detail
                detail = "You can cancel while the session starts."
            if phase == "stopping":
                detail = "Finishing recordings and releasing devices."
            if phase == "running":
                missing = next(
                    (
                        name
                        for name, at in (
                            ("Microphone", self._audio_at),
                            ("Camera", self._video_at),
                            ("Processing", self._tracks_at),
                            ("Audio output", self._playback_at),
                        )
                        if not self._fresh(at)
                    ),
                    None,
                )
                if missing:
                    phase = "starting" if elapsed < self.stale_s else "interrupted"
                    title = (
                        "Waiting for device data..."
                        if phase == "starting"
                        else f"{missing} interrupted"
                    )
                    detail = "Waiting for fresh data. Stop and retry if this persists."
                elif self._lost:
                    phase, title = "target_lost", f"{person} is out of view"
                    detail = "Select a visible person again. Audio is muted."
                elif self.mode != "live":
                    phase = "listening"
                    title = f"{person} selected" if person else "Select someone to hear"
                    detail = "Preview only. Live isolation unavailable; output muted."
                elif self._backend_unavailable:
                    phase, title = "unavailable", "Voice isolation unavailable"
                    detail = (
                        "The isolation model is not producing real output. "
                        "Stop and retry; audio is muted."
                    )
                elif self._requested is None:
                    phase, title = "listening", "Select someone to hear"
                    detail = "Camera and microphone are receiving data."
                elif self._output_ready:
                    phase, title = "isolating", f"Hearing {person}"
                    detail = "Processed audio for your selection is arriving."
                else:
                    phase, title = "focusing", f"Focusing on {person}..."
                    detail = "Waiting for processed audio. Output is muted."
            return SessionView(
                phase,
                title,
                detail,
                self.mode,
                self._phase in ("starting", "running", "stopping"),
                self._requested,
                person,
                self._input_level if self._fresh(self._audio_at) else 0.0,
                self._output_level if phase == "isolating" else 0.0,
                elapsed if self._phase == "running" else 0.0,
                len(self._visible) if self._fresh(self._tracks_at) else 0,
            )
