

from __future__ import annotations

import array
import json
import math
import queue
import sys
import tempfile
import threading
import time
import wave
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO


@dataclass(frozen=True)
class RecordingSnapshot:
    active: bool
    elapsed_s: float
    path: Path | None
    error: str | None

@dataclass
class _Packet:
    stream: str
    samples: array.array
    sample_rate: int
    channels: int
    source_timestamp_ms: float
    arrival_offset_ms: float
    metadata: dict[str, Any]

@dataclass
class _Clip:
    path: Path
    mode: str
    started_at: float
    started_utc: str
    packets: queue.Queue[_Packet]
    stop: threading.Event = field(default_factory=threading.Event)
    active: bool = True
    stopped_at: float | None = None
    error: str | None = None
    thread: threading.Thread | None = None

class SessionRecorder:

    def __init__(self, max_queue_chunks: int = 256) -> None:
        if max_queue_chunks < 1:
            raise ValueError("max_queue_chunks must be positive")
        self._capacity = max_queue_chunks
        self._lock = threading.Lock()
        self._operations = threading.Lock()
        self._clip: _Clip | None = None
        self._start_error: str | None = None

    def start_clip(self, root: Path, mode: str) -> Path:

        with self._operations:
            with self._lock:
                if self._clip is not None and self._clip.active:
                    return self._clip.path
            self._stop_clip()
            with self._lock:
                self._clip = None
                self._start_error = None
            try:
                root = Path(root)
                root.mkdir(parents=True, exist_ok=True)
                stamp = datetime.now(timezone.utc)
                path = Path(
                    tempfile.mkdtemp(
                        prefix=stamp.strftime("clip_%Y%m%d_%H%M%S_"), dir=root
                    )
                )
                clip = _Clip(
                    path,
                    str(mode),
                    time.monotonic(),
                    stamp.isoformat(),
                    queue.Queue(maxsize=self._capacity),
                )
                clip.thread = threading.Thread(
                    target=self._run,
                    args=(clip,),
                    name="clip-recorder",
                    daemon=True,
                )
                with self._lock:
                    self._clip = clip
                clip.thread.start()
                return path
            except Exception as exc:
                with self._lock:
                    self._start_error = f"Could not start recording: {exc}"
                    if self._clip is not None:
                        self._clip.active = False
                        self._clip.stopped_at = time.monotonic()
                        self._clip.error = self._start_error
                raise

    def stop_clip(self) -> Path | None:
        with self._operations:
            return self._stop_clip()

    def _stop_clip(self) -> Path | None:
        with self._lock:
            clip = self._clip
            if clip is None:
                return None
            self._end_locked(clip)
        if clip.thread is not None and clip.thread.ident is not None:
            clip.thread.join()
        return clip.path

    def snapshot(self) -> RecordingSnapshot:
        with self._lock:
            clip = self._clip
            if clip is None:
                return RecordingSnapshot(False, 0.0, None, self._start_error)
            end = clip.stopped_at if clip.stopped_at is not None else time.monotonic()
            return RecordingSnapshot(
                clip.active,
                max(0.0, end - clip.started_at),
                clip.path,
                clip.error,
            )

    def on_input(self, chunk: Any) -> None:
        self._enqueue("input", chunk)

    def on_output(self, chunk: Any) -> None:
        self._enqueue("output", chunk)

    def _end_locked(self, clip: _Clip, error: str | None = None) -> None:
        clip.active = False
        if clip.stopped_at is None:
            clip.stopped_at = time.monotonic()
        if error and clip.error is None:
            clip.error = error
        clip.stop.set()

    def _enqueue(self, stream: str, chunk: Any) -> None:
        with self._lock:
            clip = self._clip
            if clip is None or not clip.active:
                return
        try:
            data = chunk.data
            if hasattr(data, "reshape"):
                data = data.reshape(-1)
            samples = array.array("f", data)
            rate, channels = int(chunk.sample_rate), int(chunk.channels)
            timestamp = float(chunk.timestamp_ms)
            if rate <= 0 or channels <= 0 or len(samples) % channels:
                raise ValueError("Invalid audio sample rate, channels, or frame length")
            if not math.isfinite(timestamp):
                raise ValueError("Invalid audio timestamp")
            if not samples:
                return
            keys = (
                "backend",
                "conditioning",
                "target_track_id",
                "fallback",
                "no_target",
                "passthrough",
                "ui_selection_epoch",
                "ui_muted",
                "output_ready",
                "input_device",
                "input_device_requested",
                "overflowed",
                "rumble_filter_hz",
                "noise_suppression",
                "denoise_delay_samples",
                "denoise_delay_ms",
                "lip_gate_gain_min",
                "lip_gate_known_fraction",
                "source_timestamp_ms",
                "buffered_delay_ms",
                "visual_coverage",
                "visual_max_age_ms",
                "dropped_hops",
                "audio_discontinuities",
            )
            metadata = {
                key: value
                for key in keys
                if isinstance((value := chunk.metadata.get(key)), (str, bool, int))
                or (isinstance(value, float) and math.isfinite(value))
                or (value is None and key in chunk.metadata)
            }
            packet = _Packet(
                stream,
                samples,
                rate,
                channels,
                timestamp,
                (time.monotonic() - clip.started_at) * 1000,
                metadata,
            )
            with self._lock:
                if self._clip is clip and clip.active:
                    try:
                        clip.packets.put_nowait(packet)
                    except queue.Full:
                        self._end_locked(
                            clip,
                            "Recording stopped: disk writer queue is full; "
                            "clip is incomplete.",
                        )
        except Exception as exc:
            with self._lock:
                self._end_locked(clip, f"Recording stopped: {exc}")

    def _write_packet(
        self,
        clip: _Clip,
        packet: _Packet,
        writers: dict[str, tuple[wave.Wave_write, BinaryIO]],
        streams: dict[str, dict[str, Any]],
    ) -> None:
        stream = packet.stream
        if stream not in writers:
            filename = f"{stream}_audio.wav"
            handle = (clip.path / filename).open("xb")
            try:
                writer = wave.open(handle, "wb")
                writer.setparams(
                    (
                        packet.channels,
                        2,
                        packet.sample_rate,
                        0,
                        "NONE",
                        "not compressed",
                    )
                )
            except Exception:
                handle.close()
                raise
            writers[stream] = (writer, handle)
            streams[stream] = {
                "file": filename,
                "sample_rate": packet.sample_rate,
                "channels": packet.channels,
                "sample_width_bytes": 2,
                "frames_written": 0,
                "chunks": [],
            }
        info = streams[stream]
        if (info["sample_rate"], info["channels"]) != (
            packet.sample_rate,
            packet.channels,
        ):
            raise ValueError(f"{stream} audio format changed during the clip")
        pcm = array.array(
            "h",
            (max(-32768, min(32767, int(sample * 32767))) for sample in packet.samples),
        )
        if sys.byteorder != "little":
            pcm.byteswap()
        writers[stream][0].writeframes(pcm.tobytes())
        frames = len(packet.samples) // packet.channels
        info["chunks"].append(
            {
                "wav_frame_offset": info["frames_written"],
                "frame_count": frames,
                "source_timestamp_ms": packet.source_timestamp_ms,
                "arrival_offset_ms": packet.arrival_offset_ms,
                "metadata": packet.metadata,
            }
        )
        info["frames_written"] += frames

    def _run(self, clip: _Clip) -> None:
        writers: dict[str, tuple[wave.Wave_write, BinaryIO]] = {}
        streams: dict[str, dict[str, Any]] = {}
        try:
            while not clip.stop.is_set() or not clip.packets.empty():
                try:
                    packet = clip.packets.get(timeout=0.05)
                except queue.Empty:
                    continue
                self._write_packet(clip, packet, writers, streams)
        except Exception as exc:
            with self._lock:
                self._end_locked(clip, f"Recording stopped: {exc}")
        finally:
            for writer, handle in writers.values():
                try:
                    writer.close()
                except Exception as exc:
                    with self._lock:
                        self._end_locked(
                            clip, f"Could not finalize recorded audio: {exc}"
                        )
                finally:
                    try:
                        handle.close()
                    except Exception as exc:
                        with self._lock:
                            self._end_locked(
                                clip, f"Could not close recorded audio: {exc}"
                            )
            with self._lock:
                self._end_locked(clip)
                duration = max(
                    0.0, (clip.stopped_at or clip.started_at) - clip.started_at
                )
                error = clip.error
            preview = any(
                name in clip.mode.lower() for name in ("preview", "passthrough", "mock")
            )
            manifest = {
                "schema_version": 1,
                "mode": clip.mode,
                "started_utc": clip.started_utc,
                "elapsed_s": duration,
                "status": "incomplete" if error else "completed",
                "error": error,
                "output_description": (
                    "Preview or passthrough audio; not evidence of speech isolation."
                    if preview
                    else "Pipeline output; isolation quality is not verified."
                ),
                "timeline": (
                    "Each WAV concatenates received frames without gap padding. "
                    "Use per-chunk source timestamps and arrival offsets to "
                    "reconstruct timing. No video is recorded."
                ),
                "streams": streams,
            }
            try:
                with (clip.path / "manifest.json").open(
                    "x", encoding="utf-8"
                ) as manifest_handle:
                    json.dump(manifest, manifest_handle, indent=2, allow_nan=False)
            except Exception as exc:
                with self._lock:
                    self._end_locked(clip, f"Could not save recording manifest: {exc}")
