

from __future__ import annotations

import array
import logging
import threading
import time
from collections import deque

from onevoice.core.models.audio_chunk import AudioChunk
from onevoice.core.models.frame import Frame

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is a core dependency elsewhere
    np = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

def _concat_audio_data(datas: list) -> object:

    first = datas[0]
    if np is not None and isinstance(first, np.ndarray):
        return np.concatenate(datas, axis=-1)
    if isinstance(first, array.array):
        merged = array.array(first.typecode)
        for d in datas:
            merged.extend(d)
        return merged
    merged_list: list = []
    for d in datas:
        merged_list.extend(d)
    return merged_list

def _merge_audio_chunks(chunks: list[AudioChunk]) -> AudioChunk:

    if len(chunks) == 1:
        return chunks[0]
    first = chunks[0]
    return AudioChunk(
        timestamp_ms=first.timestamp_ms,
        data=_concat_audio_data([c.data for c in chunks]),
        sample_rate=first.sample_rate,
        channels=first.channels,
        metadata=first.metadata,
    )

class TimestampFrameSynchronizer:

    def __init__(
        self,
        sync_tolerance_ms: float = 40.0,
        max_buffer_size: int = 32,
        max_drift_ms: float = 200.0,
        pair_timeout_s: float = 0.1,
        max_audio_backlog_ms: float = 3000.0,
        frame_hold_ms: float = 500.0,
        audio_release_ms: float = 100.0,
    ) -> None:
        self._sync_tolerance_ms = sync_tolerance_ms
        self._max_buffer_size = max_buffer_size
        self._max_drift_ms = max_drift_ms
        self._pair_timeout_s = pair_timeout_s
        self._max_audio_backlog_ms = max_audio_backlog_ms
        self._frame_hold_ms = frame_hold_ms
        self._audio_release_ms = audio_release_ms

        self._audio_buffer: deque[AudioChunk] = deque()
        self._video_buffer: deque[Frame] = deque(maxlen=max_buffer_size)
        self._last_frame: Frame | None = None
        self._lock = threading.Lock()
        self._pair_ready = threading.Condition(self._lock)
        self._dropped_audio = 0
        self._dropped_video = 0
        self._held_frames = 0
        self._last_drift_ms = 0.0
        self._max_observed_drift_ms = 0.0

    @property
    def dropped_audio(self) -> int:
        return self._dropped_audio

    @property
    def dropped_video(self) -> int:
        return self._dropped_video

    @property
    def held_frames(self) -> int:

        return self._held_frames

    @property
    def audio_backlog(self) -> int:
        with self._lock:
            return len(self._audio_buffer)

    @property
    def last_drift_ms(self) -> float:
        return self._last_drift_ms

    @property
    def max_drift_ms(self) -> float:
        return self._max_observed_drift_ms

    def push_audio(self, chunk: AudioChunk) -> None:
        with self._pair_ready:
            self._audio_buffer.append(chunk)
            self._trim_drift_locked()
            self._trim_backlog_locked()
            self._pair_ready.notify_all()

    def push_video(self, frame: Frame) -> None:
        with self._pair_ready:
            self._video_buffer.append(frame)
            self._trim_drift_locked()
            self._pair_ready.notify_all()

    def get_synchronized_pair(self) -> tuple[AudioChunk, Frame]:

        chunk, frame, _held = self.get_pair_with_status()
        return chunk, frame

    def get_pair_with_status(self) -> tuple[AudioChunk, Frame, bool]:

        deadline = time.monotonic() + self._pair_timeout_s
        with self._pair_ready:
            while True:
                pair = self._try_match_locked()
                if pair is not None:
                    return pair
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Timed out waiting for synchronized A/V pair")
                self._pair_ready.wait(timeout=remaining)

    def _backlog_ms_locked(self) -> float:
        if len(self._audio_buffer) < 2:
            return 0.0
        return float(
            self._audio_buffer[-1].timestamp_ms - self._audio_buffer[0].timestamp_ms
        )

    def _trim_drift_locked(self) -> None:

        while (
            self._audio_buffer
            and self._video_buffer
            and abs(
                self._audio_buffer[0].timestamp_ms - self._video_buffer[0].timestamp_ms
            )
            > self._max_drift_ms
        ):
            if self._audio_buffer[0].timestamp_ms < self._video_buffer[0].timestamp_ms:

                break
            self._video_buffer.popleft()
            self._dropped_video += 1

    def _trim_backlog_locked(self) -> None:

        while (
            len(self._audio_buffer) > 1
            and self._backlog_ms_locked() > self._max_audio_backlog_ms
        ):
            self._audio_buffer.popleft()
            self._dropped_audio += 1
            if self._dropped_audio == 1 or self._dropped_audio % 50 == 0:
                logger.warning(
                    "Audio backlog exceeded %.0f ms with no video to pair against; "
                    "dropped %d chunk(s). The camera is not delivering frames.",
                    self._max_audio_backlog_ms,
                    self._dropped_audio,
                )

    def _drain_audio_locked(self, upper_ms: float | None) -> list[AudioChunk]:
        matched: list[AudioChunk] = []
        while self._audio_buffer and (
            upper_ms is None or self._audio_buffer[0].timestamp_ms <= upper_ms
        ):
            matched.append(self._audio_buffer.popleft())
        return matched

    def _try_match_locked(self) -> tuple[AudioChunk, Frame, bool] | None:
        while self._audio_buffer and self._video_buffer:
            video_ts = self._video_buffer[0].timestamp_ms
            upper = video_ts + self._sync_tolerance_ms

            if self._audio_buffer[0].timestamp_ms > upper:

                self._video_buffer.popleft()
                self._dropped_video += 1
                continue

            matched = self._drain_audio_locked(upper)
            frame = self._video_buffer.popleft()
            self._last_frame = frame
            merged = _merge_audio_chunks(matched)
            delta = abs(merged.timestamp_ms - frame.timestamp_ms)
            self._last_drift_ms = delta
            self._max_observed_drift_ms = max(self._max_observed_drift_ms, delta)
            return merged, frame, False

        if (
            self._audio_buffer
            and not self._video_buffer
            and self._last_frame is not None
            and self._backlog_ms_locked() >= self._audio_release_ms
        ):
            age = self._audio_buffer[-1].timestamp_ms - self._last_frame.timestamp_ms
            if age <= self._frame_hold_ms:
                matched = self._drain_audio_locked(None)
                self._held_frames += 1
                return _merge_audio_chunks(matched), self._last_frame, True

        return None
