

from __future__ import annotations

import array
import logging
import queue
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np

from onevoice.core.models.audio_chunk import AudioChunk

logger = logging.getLogger("onevoice.transcription")

class Transcriber(Protocol):

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str: ...

class FakeTranscriber:

    def __init__(self, text: str = "fake transcript") -> None:
        self.text = text
        self.calls: list[tuple[int, int]] = []
        self._done = threading.Event()

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        self.calls.append((len(audio), sample_rate))
        self._done.set()
        return self.text

    def wait_for_call(self, timeout: float = 2.0) -> bool:

        return self._done.wait(timeout)

    def reset(self) -> None:
        self._done.clear()

class FasterWhisperTranscriber:

    def __init__(
        self,
        model_size: str = "large-v3",
        compute_type: str = "int8",
        device: str = "cpu",
    ) -> None:
        from faster_whisper import WhisperModel

        self._model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        if sample_rate != 16_000:
            raise ValueError(
                f"FasterWhisperTranscriber assumes 16kHz input (Whisper's native "
                f"rate); got {sample_rate}. Resample before calling this."
            )

        segments, _info = self._model.transcribe(audio, beam_size=5, vad_filter=True)
        return " ".join(segment.text.strip() for segment in segments)

def _chunk_to_float32(chunk: AudioChunk) -> np.ndarray:

    data = chunk.data
    if isinstance(data, array.array):
        return np.frombuffer(data, dtype=np.float32)
    if hasattr(data, "astype"):
        return np.asarray(data, dtype=np.float32).reshape(-1)
    return np.array(data, dtype=np.float32)

class TranscribingSink:

    def __init__(
        self,
        label: str,
        transcriber: Transcriber,
        on_transcript: Callable[[str, str, float], None],
        sample_rate: int = 16_000,
        max_segment_s: float = 6.0,
        silence_thresh: float | None = 0.02,
        silence_hold_s: float = 0.6,
        min_segment_s: float = 0.0,
    ) -> None:
        self._label = label
        self._transcriber = transcriber
        self._on_transcript = on_transcript
        self._sample_rate = sample_rate
        self._max_segment_samples = int(max_segment_s * sample_rate)
        self._silence_thresh = silence_thresh
        self._silence_hold_s = silence_hold_s
        self._min_segment_samples = int(min_segment_s * sample_rate)

        self._lock = threading.Lock()
        self._buffer: list[np.ndarray] = []
        self._buffered_samples = 0
        self._buffer_start_ms: float | None = None
        self._has_speech = False
        self._silence_accum_s = 0.0

        self._queue: queue.Queue[tuple[np.ndarray, float] | None] = queue.Queue()
        self._worker: threading.Thread | None = None

    def start(self) -> None:
        with self._lock:
            self._reset_buffer_locked()
        self._worker = threading.Thread(
            target=self._worker_loop, name=f"transcribe-{self._label}", daemon=True
        )
        self._worker.start()

    def stop(self) -> None:
        self._flush()
        self._queue.put(None)
        if self._worker is not None:
            self._worker.join(timeout=10.0)
            self._worker = None

    def write(self, chunk: AudioChunk) -> None:
        samples = _chunk_to_float32(chunk)
        n = len(samples)
        if n == 0:
            return
        vad_on = self._silence_thresh is not None
        is_silent = vad_on and float(np.sqrt(np.mean(np.square(samples)))) < (
            self._silence_thresh or 0.0
        )
        duration_s = n / self._sample_rate if self._sample_rate else 0.0

        with self._lock:
            if vad_on and not self._has_speech and is_silent:
                return
            if self._buffer_start_ms is None:
                self._buffer_start_ms = chunk.timestamp_ms
            self._buffer.append(samples)
            self._buffered_samples += n
            if vad_on:
                if is_silent:
                    self._silence_accum_s += duration_s
                else:
                    self._has_speech = True
                    self._silence_accum_s = 0.0

            ready = self._buffered_samples >= self._max_segment_samples
            if (
                vad_on
                and self._has_speech
                and self._silence_accum_s >= self._silence_hold_s
                and self._buffered_samples >= self._min_segment_samples
            ):
                ready = True
        if ready:
            self._flush()

    def _reset_buffer_locked(self) -> None:

        self._buffer = []
        self._buffered_samples = 0
        self._buffer_start_ms = None
        self._has_speech = False
        self._silence_accum_s = 0.0

    def _flush(self) -> None:

        with self._lock:
            if not self._buffer:
                return
            segment = np.concatenate(self._buffer)
            start_ms = self._buffer_start_ms
            if start_ms is None:
                start_ms = 0.0
            self._reset_buffer_locked()
        self._queue.put((segment, start_ms))

    def _worker_loop(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                break
            segment, start_ms = item
            try:
                text = self._transcriber.transcribe(segment, self._sample_rate)
            except (
                Exception
            ) as exc:  # noqa: BLE001 - one bad segment must not kill the worker
                text = f"[transcription error: {exc}]"
            if not text.strip():
                continue
            try:
                self._on_transcript(self._label, text.strip(), start_ms)
            except (
                Exception
            ):  # noqa: BLE001 - a bad callback (e.g. a console print hitting

                logger.exception(
                    "on_transcript callback failed for label=%r -- transcript "
                    "dropped, worker continues",
                    self._label,
                )
