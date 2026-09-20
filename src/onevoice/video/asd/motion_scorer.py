

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

from onevoice.core.models.speaker_track import SpeakerTrack

@dataclass(frozen=True)
class AsdConfig:
    history_len: int = 8
    motion_scale: float = 8.0

class MotionActiveSpeakerScorer:

    def __init__(self, config: AsdConfig | None = None) -> None:
        self._config = config or AsdConfig()
        self._history: dict[str, deque[tuple[float, float]]] = {}

    def score(self, track: SpeakerTrack) -> float:
        landmarks = track.metadata.get("landmarks")
        mouth = _mouth_point(track, landmarks)
        if mouth is None:
            return 0.0
        mx, my = float(mouth[0]), float(mouth[1])
        hist = self._history.setdefault(
            track.track_id, deque(maxlen=self._config.history_len)
        )
        if hist:
            px, py = hist[-1]
            delta = math.hypot(mx - px, my - py)
            score = min(1.0, delta / max(self._config.motion_scale, 1e-6))
        else:
            score = 0.0
        hist.append((mx, my))
        return score

    def evict(self, track_id: str) -> None:
        self._history.pop(track_id, None)

def _mouth_point(
    track: SpeakerTrack, landmarks: object
) -> tuple[float, float] | None:
    if isinstance(landmarks, dict):
        for key in ("mouth", "mouth_center", "mouthCenter", "Mouth"):
            pt = landmarks.get(key)
            if pt is not None:
                return float(pt[0]), float(pt[1])
    lip = track.metadata.get("lip_roi")
    if isinstance(lip, (tuple, list)) and len(lip) >= 4:
        x, y, w, h = lip[0], lip[1], lip[2], lip[3]
        return float(x) + float(w) / 2.0, float(y) + float(h) / 2.0
    return None
