

from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from onevoice.core.models.frame import Frame
from onevoice.core.models.speaker_track import SpeakerTrack

class TrackObservingSelector:

    def __init__(self, inner: Any, smoothing_alpha: float = 0.35) -> None:
        self._inner = inner
        self._lock = threading.Lock()
        self._latest_tracks: list[SpeakerTrack] = []
        self._smoothing_alpha = smoothing_alpha
        self._smoothed_boxes: dict[str, tuple[float, float, float, float]] = {}

    def _smoothed(self, tracks: list[SpeakerTrack]) -> list[SpeakerTrack]:
        alpha = self._smoothing_alpha
        seen: set[str] = set()
        out: list[SpeakerTrack] = []
        for t in tracks:
            seen.add(t.track_id)
            prev = self._smoothed_boxes.get(t.track_id)
            if prev is None:
                box = t.bounding_box
            else:
                box = (
                    prev[0] + alpha * (t.bounding_box[0] - prev[0]),
                    prev[1] + alpha * (t.bounding_box[1] - prev[1]),
                    prev[2] + alpha * (t.bounding_box[2] - prev[2]),
                    prev[3] + alpha * (t.bounding_box[3] - prev[3]),
                )
            self._smoothed_boxes[t.track_id] = box
            out.append(
                SpeakerTrack(
                    track_id=t.track_id,
                    bounding_box=box,
                    confidence=t.confidence,
                    metadata=t.metadata,
                )
            )

        for stale_id in set(self._smoothed_boxes) - seen:
            del self._smoothed_boxes[stale_id]
        return out

    def select_target(self, frame: Frame, tracks: list[SpeakerTrack]) -> Any:
        with self._lock:
            self._latest_tracks = self._smoothed(tracks)
        return self._inner.select_target(frame, tracks)

    def latest_tracks(self) -> list[SpeakerTrack]:
        with self._lock:
            return list(self._latest_tracks)

    def set_manual_target(self, track_id: str | None) -> None:

        set_manual = getattr(self._inner, "set_manual_target", None)
        if callable(set_manual):
            set_manual(track_id)

    def get_state(self) -> Any:
        get_state = getattr(self._inner, "get_state", None)
        return get_state() if callable(get_state) else None

def hit_test(tracks: list[SpeakerTrack], px: float, py: float) -> str | None:

    hits = [
        t
        for t in tracks
        if t.bounding_box[0] <= px <= t.bounding_box[0] + t.bounding_box[2]
        and t.bounding_box[1] <= py <= t.bounding_box[1] + t.bounding_box[3]
    ]
    if not hits:
        return None

    smallest = min(hits, key=lambda t: t.bounding_box[2] * t.bounding_box[3])
    return smallest.track_id

def window_click_to_native(
    x: float,
    y: float,
    displayed_w: float,
    displayed_h: float,
    native_w: float,
    native_h: float,
) -> tuple[float, float]:

    if displayed_w <= 0 or displayed_h <= 0:
        return x, y
    return x * native_w / displayed_w, y * native_h / displayed_h
