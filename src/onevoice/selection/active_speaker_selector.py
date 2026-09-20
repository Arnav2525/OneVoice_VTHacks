

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from enum import Enum

from onevoice.core.models.frame import Frame
from onevoice.core.models.speaker_track import SpeakerTrack
from onevoice.core.models.target_selection import TargetSelection

logger = logging.getLogger(__name__)

class LockState(str, Enum):
    IDLE = "idle"
    LOCKED = "locked"
    COASTING = "coasting"
    SEARCHING = "searching"

@dataclass(frozen=True)
class SelectorConfig:
    acquire_thresh: float = 0.55
    acquire_dwell_ms: float = 200.0
    switch_margin: float = 0.15
    switch_dwell_ms: float = 400.0
    coast_ms: float = 500.0

class ActiveSpeakerSelector:

    def __init__(self, config: SelectorConfig | None = None) -> None:
        self._config = config or SelectorConfig()
        self._state = LockState.IDLE
        self._lock_id: str | None = None
        self._last_track: SpeakerTrack | None = None
        self._lock_since_ms = 0.0
        self._candidate_id: str | None = None
        self._candidate_since_ms = 0.0
        self._switch_candidate_id: str | None = None
        self._switch_since_ms = 0.0
        self._coast_since_ms = 0.0
        self._manual_id: str | None = None
        self._manual_last_track: SpeakerTrack | None = None
        self._manual_lost_since_ms: float | None = None
        self._lock = threading.Lock()

    def set_manual_target(self, track_id: str | None) -> None:
        with self._lock:
            if track_id != self._manual_id:

                self._manual_last_track = None
                self._manual_lost_since_ms = None
            self._manual_id = track_id
            if track_id is not None:
                self._state = LockState.LOCKED
                self._lock_id = track_id

    def get_state(self) -> LockState:
        with self._lock:
            return self._state

    def select_target(
        self, frame: Frame, tracks: list[SpeakerTrack]
    ) -> TargetSelection:
        now = frame.timestamp_ms
        with self._lock:
            manual = self._manual_id
        if manual is not None:
            found = _find_track(tracks, manual)
            if found is not None:
                self._manual_last_track = found
                self._manual_lost_since_ms = None
                return TargetSelection(timestamp_ms=now, selected_speaker=found)

            if self._manual_lost_since_ms is None:
                self._manual_lost_since_ms = now
            if (
                self._manual_last_track is not None
                and now - self._manual_lost_since_ms <= self._config.coast_ms
            ):
                return TargetSelection(
                    timestamp_ms=now, selected_speaker=self._manual_last_track
                )
            return TargetSelection(timestamp_ms=now, selected_speaker=None)

        best = _best_speaking(tracks)

        if self._state is LockState.IDLE:
            if best and best[1] >= self._config.acquire_thresh:
                if self._candidate_id != best[0].track_id:
                    self._candidate_id = best[0].track_id
                    self._candidate_since_ms = now
                    if self._config.acquire_dwell_ms <= 0:
                        self._engage(best[0], now)
                elif now - self._candidate_since_ms >= self._config.acquire_dwell_ms:
                    self._engage(best[0], now)
            else:
                self._candidate_id = None
            locked = _find_track(tracks, self._lock_id)
            return self._emit(now, locked)

        locked = _find_track(tracks, self._lock_id)

        if self._state is LockState.LOCKED:
            if locked is not None:
                self._last_track = locked
                self._maybe_switch(locked, tracks, now)
                locked = _find_track(tracks, self._lock_id)
                return self._emit(now, locked)
            if self._last_track is not None:
                self._state = LockState.COASTING
                self._coast_since_ms = now
                return self._emit(now, self._last_track, coasting=True)
            self._state = LockState.SEARCHING
            self._lock_id = None
            return self._emit(now, None)

        if self._state is LockState.COASTING:
            if locked is not None:
                self._state = LockState.LOCKED
                self._last_track = locked
                return self._emit(now, locked)
            if (
                self._last_track is not None
                and now - self._coast_since_ms <= self._config.coast_ms
            ):
                return self._emit(now, self._last_track, coasting=True)
            self._state = LockState.SEARCHING
            self._lock_id = None
            return self._emit(now, None)

        if best and best[1] >= self._config.acquire_thresh:
            if self._candidate_id != best[0].track_id:
                self._candidate_id = best[0].track_id
                self._candidate_since_ms = now
                if self._config.acquire_dwell_ms <= 0:
                    self._engage(best[0], now)
                    return self._emit(now, best[0])
            elif now - self._candidate_since_ms >= self._config.acquire_dwell_ms:
                self._engage(best[0], now)
                return self._emit(now, best[0])
        return self._emit(now, None)

    def _engage(self, track: SpeakerTrack, now: float) -> None:
        if self._lock_id != track.track_id:
            logger.info("active-speaker lock acquired: %s", track.track_id)
        self._state = LockState.LOCKED
        self._lock_id = track.track_id
        self._last_track = track
        self._lock_since_ms = now
        self._coast_since_ms = 0.0
        self._switch_candidate_id = None

    def _maybe_switch(
        self,
        locked: SpeakerTrack,
        tracks: list[SpeakerTrack],
        now: float,
    ) -> None:
        best = _best_speaking(tracks)
        if best is None or best[0].track_id == locked.track_id:
            self._switch_candidate_id = None
            return
        locked_score = float(locked.metadata.get("speaking_score", 0.0))
        margin = best[1] - locked_score
        if margin < self._config.switch_margin:
            self._switch_candidate_id = None
            return
        if self._switch_candidate_id != best[0].track_id:
            self._switch_candidate_id = best[0].track_id
            self._switch_since_ms = now
        elif now - self._switch_since_ms >= self._config.switch_dwell_ms:
            logger.info(
                "active-speaker switch %s -> %s", locked.track_id, best[0].track_id
            )
            self._engage(best[0], now)

    def _emit(
        self,
        now: float,
        track: SpeakerTrack | None,
        coasting: bool = False,
    ) -> TargetSelection:
        if track is not None:
            meta = dict(track.metadata)
            meta["lock_state"] = self._state.value
            meta["coasting"] = coasting
            track = SpeakerTrack(
                track_id=track.track_id,
                bounding_box=track.bounding_box,
                confidence=track.confidence,
                metadata=meta,
            )
        return TargetSelection(timestamp_ms=now, selected_speaker=track)

def _find_track(
    tracks: list[SpeakerTrack], track_id: str | None
) -> SpeakerTrack | None:
    if track_id is None:
        return None
    for t in tracks:
        if t.track_id == track_id:
            return t
    return None

def _best_speaking(
    tracks: list[SpeakerTrack],
) -> tuple[SpeakerTrack, float] | None:
    best: SpeakerTrack | None = None
    best_score = -1.0
    for t in tracks:
        score = float(t.metadata.get("speaking_score", 0.0))
        if score > best_score:
            best_score = score
            best = t
    if best is None:
        return None
    return best, best_score
