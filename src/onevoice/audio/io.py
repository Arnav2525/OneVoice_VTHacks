

from __future__ import annotations

import array
import logging
import threading
import time
import wave
from pathlib import Path
from typing import Any

from onevoice.audio.ring_buffer import RingBuffer
from onevoice.core.models.audio_chunk import AudioChunk

logger = logging.getLogger(__name__)

try:
    import sounddevice as sd
except (ImportError, OSError):  # pragma: no cover - optional / no PortAudio (Kaggle)

    sd = None  # type: ignore[assignment,misc]

try:

    import numpy as np
except ImportError:  # pragma: no cover - optional runtime dependency
    np = None  # type: ignore[assignment]

def _now_ms() -> float:
    return time.monotonic() * 1000.0

def _stop_all(devices: list[Any]) -> None:

    first_error = None
    for device in devices:
        try:
            device.stop()
        except Exception as exc:
            logger.exception("Failed to stop %s", type(device).__name__)
            first_error = first_error or exc
    if first_error is not None:
        raise first_error

class MicrophoneSource:

    def __init__(
        self,
        sample_rate: int = 16_000,
        channels: int = 1,
        chunk_samples: int = 320,
        device: int | None = None,
    ) -> None:
        self._sample_rate = sample_rate
        self._channels = channels
        self._chunk_samples = chunk_samples
        self._device = device
        self._stream: Any = None
        self._running = False
        self._lock = threading.Lock()

    def start(self) -> None:
        if sd is None:
            raise RuntimeError(
                "sounddevice + PortAudio are required for MicrophoneSource; "
                "pip install sounddevice and install the PortAudio system library"
            )
        with self._lock:
            if self._running:
                return
            self._stream = sd.InputStream(
                samplerate=self._sample_rate,
                channels=self._channels,
                dtype="float32",
                blocksize=self._chunk_samples,
                device=self._device,
            )
            self._stream.start()
            self._running = True

    def stop(self) -> None:
        with self._lock:
            self._running = False
            if self._stream is not None:
                stop_error = None
                try:
                    self._stream.stop()
                except Exception as exc:
                    stop_error = exc
                try:
                    self._stream.close()
                    self._stream = None
                except Exception:
                    if stop_error is None:
                        raise
                    logger.exception("Microphone close also failed after stop failed")
                if stop_error is not None:
                    raise stop_error

    def read(self) -> AudioChunk:
        if not self._running or self._stream is None:
            raise RuntimeError("MicrophoneSource is not running")
        capture_start = _now_ms()
        data, _overflowed = self._stream.read(self._chunk_samples)

        flat = np.ascontiguousarray(data, dtype=np.float32).reshape(-1)
        samples = array.array("f")
        samples.frombytes(flat.tobytes())
        return AudioChunk(
            timestamp_ms=capture_start,
            data=samples,
            sample_rate=self._sample_rate,
            channels=self._channels,
            metadata={"overflowed": bool(_overflowed)},
        )

class SpeakerSink:

    def __init__(
        self,
        sample_rate: int = 16_000,
        channels: int = 1,
        buffer_samples: int = 1_600,
        device: int | None = None,
    ) -> None:
        self._sample_rate = sample_rate
        self._channels = channels
        self._device = device
        self._ring = RingBuffer(buffer_samples)
        self._stream: Any = None
        self._running = False
        self._lock = threading.Lock()
        self._playback_block = 320
        self._underruns = 0

    @property
    def underruns(self) -> int:
        return self._underruns

    def start(self) -> None:
        if sd is None:
            raise RuntimeError(
                "sounddevice + PortAudio are required for SpeakerSink; "
                "pip install sounddevice and install the PortAudio system library"
            )
        with self._lock:
            if self._running:
                return

            def callback(
                outdata: Any,
                frames: int,
                _time_info: Any,
                _status: Any,
            ) -> None:
                needed = frames * self._channels
                samples = self._ring.read(needed, timeout=0.01)
                outdata.fill(0)
                limit = min(len(samples), needed)
                if limit < needed:
                    self._underruns += 1
                if limit:

                    n_frames = limit // self._channels
                    flat = np.frombuffer(samples, dtype=np.float32, count=limit)
                    outdata[:n_frames] = flat[: n_frames * self._channels].reshape(
                        n_frames, self._channels
                    )

            self._stream = sd.OutputStream(
                samplerate=self._sample_rate,
                channels=self._channels,
                dtype="float32",
                blocksize=self._playback_block,
                device=self._device,
                callback=callback,
            )
            self._stream.start()
            self._running = True

    def stop(self) -> None:
        with self._lock:
            self._running = False
            self._ring.clear()
            if self._stream is not None:
                stop_error = None
                try:
                    self._stream.stop()
                except Exception as exc:
                    stop_error = exc
                try:
                    self._stream.close()
                    self._stream = None
                except Exception:
                    if stop_error is None:
                        raise
                    logger.exception("Speaker close also failed after stop failed")
                if stop_error is not None:
                    raise stop_error

    def write(self, chunk: AudioChunk) -> None:
        if not self._running:
            raise RuntimeError("SpeakerSink is not running")
        data = chunk.data
        if isinstance(data, array.array):
            buf = data
        elif isinstance(data, list):
            buf = array.array("f", data)
        elif hasattr(data, "tolist"):
            buf = array.array("f", (float(x) for x in data.tolist()))
        else:
            raise TypeError(
                "SpeakerSink expects array.array, list, or numpy float samples"
            )
        self._ring.write(buf, timeout=0.05)

def _chunk_to_int16_bytes(chunk: AudioChunk) -> bytes:
    data = chunk.data
    if isinstance(data, array.array):
        floats = data
    elif isinstance(data, list):
        floats = array.array("f", data)
    elif hasattr(data, "tolist"):
        floats = array.array("f", (float(x) for x in data.tolist()))
    else:
        floats = array.array("f", data)
    clamped = array.array(
        "h", (max(-32768, min(32767, int(x * 32767.0))) for x in floats)
    )
    return clamped.tobytes()

class WavFileSink:

    def __init__(self, path: str | Path, sample_rate: int, channels: int) -> None:
        self._path = Path(path)
        self._sample_rate = sample_rate
        self._channels = channels
        self._wav: wave.Wave_write | None = None
        self._lock = threading.Lock()

    def start(self) -> None:
        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._wav = wave.open(str(self._path), "wb")
            self._wav.setnchannels(self._channels)
            self._wav.setsampwidth(2)
            self._wav.setframerate(self._sample_rate)

    def stop(self) -> None:
        with self._lock:
            if self._wav is not None:
                self._wav.close()
                self._wav = None

    def write(self, chunk: AudioChunk) -> None:
        with self._lock:
            if self._wav is not None:
                self._wav.writeframes(_chunk_to_int16_bytes(chunk))

class TeeAudioSink:

    def __init__(self, sinks: list[Any]) -> None:
        self._sinks = sinks

    def start(self) -> None:
        for sink in self._sinks:
            sink.start()

    def stop(self) -> None:
        _stop_all(self._sinks)

    def write(self, chunk: AudioChunk) -> None:
        for sink in self._sinks:
            sink.write(chunk)

class RecordingAudioSource:

    def __init__(self, source: Any, recorder: WavFileSink) -> None:
        self._source = source
        self._recorder = recorder

    def start(self) -> None:
        self._source.start()
        self._recorder.start()

    def stop(self) -> None:
        _stop_all([self._source, self._recorder])

    def read(self) -> AudioChunk:
        chunk = self._source.read()
        self._recorder.write(chunk)
        return chunk

class MockAudioSource:

    def __init__(
        self,
        sample_rate: int = 16_000,
        channels: int = 1,
        chunk_samples: int = 320,
    ) -> None:
        self._sample_rate = sample_rate
        self._channels = channels
        self._chunk_samples = chunk_samples
        self._running = False

    def start(self) -> None:
        self._running = True

    def stop(self) -> None:
        self._running = False

    def read(self) -> AudioChunk:
        if not self._running:
            raise RuntimeError("MockAudioSource is not running")
        time.sleep(self._chunk_samples / self._sample_rate)
        samples = array.array("f", [0.0] * (self._chunk_samples * self._channels))
        return AudioChunk(
            timestamp_ms=_now_ms(),
            data=samples,
            sample_rate=self._sample_rate,
            channels=self._channels,
            metadata={"mock": True},
        )

class MockAudioSink:

    def __init__(self) -> None:
        self._running = False
        self.writes = 0
        self.last_chunk: AudioChunk | None = None

    def start(self) -> None:
        self._running = True

    def stop(self) -> None:
        self._running = False

    def write(self, chunk: AudioChunk) -> None:
        if not self._running:
            raise RuntimeError("MockAudioSink is not running")
        self.writes += 1
        self.last_chunk = chunk
