

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable
from typing import Protocol

from onevoice.core.interfaces import (
    AudioSink,
    AudioSource,
    FaceTracker,
    FrameSynchronizer,
    LatencyRecorder,
    TargetSelector,
    TargetSeparator,
)
from onevoice.core.models.audio_chunk import AudioChunk
from onevoice.core.models.frame import Frame
from onevoice.core.models.pipeline_result import PipelineResult
from onevoice.streaming.synchronizer import TimestampFrameSynchronizer
from onevoice.telemetry.latency_recorder import StageLatencyRecorder

logger = logging.getLogger(__name__)

def _now_ms() -> float:
    return time.monotonic() * 1000.0

class VideoReader(Protocol):

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def read(self) -> Frame: ...

class StreamingPipeline:

    def __init__(
        self,
        audio_source: AudioSource,
        audio_sink: AudioSink,
        video_source: VideoReader,
        face_tracker: FaceTracker,
        target_selector: TargetSelector,
        target_separator: TargetSeparator,
        synchronizer: FrameSynchronizer | None = None,
        latency_recorder: LatencyRecorder | None = None,
        audio_queue_size: int = 256,
        video_queue_size: int = 4,
        playback_queue_size: int = 8,
        latency_log_interval_s: float = 2.0,
        on_result: Callable[[PipelineResult], None] | None = None,
    ) -> None:
        self._audio_source = audio_source
        self._audio_sink = audio_sink
        self._video_source = video_source
        self._face_tracker = face_tracker
        self._target_selector = target_selector
        self._target_separator = target_separator
        self._synchronizer = synchronizer or TimestampFrameSynchronizer()
        self._latency = latency_recorder or StageLatencyRecorder()
        self._on_result = on_result
        self._latency_log_interval_s = latency_log_interval_s

        self._audio_queue: queue.Queue = queue.Queue(maxsize=audio_queue_size)
        self._video_queue: queue.Queue = queue.Queue(maxsize=video_queue_size)
        self._playback_queue: queue.Queue[AudioChunk] = queue.Queue(
            maxsize=playback_queue_size
        )

        self._stop_event = threading.Event()
        self._threads: list[threading.Thread] = []
        self._running = False
        self._lock = threading.Lock()

        self._lifecycle_lock = threading.Lock()
        self._devices_to_stop: list[
            tuple[str, AudioSource | AudioSink | VideoReader]
        ] = []
        self._stop_timeout_s = 2.0
        self._results_processed = 0
        self._last_tracks: list | None = None
        self._last_target = None

        self._audio_drops = 0
        self._video_drops = 0
        self._playback_drops = 0
        self._audio_backpressure_waits = 0
        self._held_frame_pairs = 0
        self._video_read_failures = 0
        self._video_reopens = 0
        self._video_max_read_failures = 30
        self._video_retry_delay_s = 0.1
        self._start_monotonic = 0.0
        self._threads_total = 0
        self._threads_alive = 0

    @property
    def latency_recorder(self) -> LatencyRecorder:
        return self._latency

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def target_separator(self) -> TargetSeparator:

        return self._target_separator

    def get_backend_status(self) -> dict[str, object]:

        separator = self._target_separator
        if hasattr(separator, "get_status"):
            return dict(separator.get_status())
        return {
            "backend": type(separator).__name__,
            "is_real_separation": None,
        }

    def _log_active_backend(self) -> None:
        status = self.get_backend_status()
        backend = status.get("backend")
        is_real = status.get("is_real_separation")
        if is_real is True:
            logger.info("Active separation backend: '%s' (real separation)", backend)
        elif is_real is False:
            logger.warning(
                "Active separation backend: '%s' -- NO REAL SEPARATION: audio is "
                "passed through UNMODIFIED (passthrough/fallback), not separated.",
                backend,
            )
        else:
            logger.info(
                "Active separation backend: '%s' (separation status unknown)", backend
            )

    def get_stats(self) -> dict[str, object]:

        runtime = 0.0
        if self._start_monotonic:
            runtime = time.monotonic() - self._start_monotonic
        if self._threads:
            total = len(self._threads)
            alive = sum(1 for t in self._threads if t.is_alive())
        else:
            total = self._threads_total
            alive = self._threads_alive
        stats: dict[str, object] = {
            "results_processed": self._results_processed,
            "audio_drops": self._audio_drops,
            "video_drops": self._video_drops,
            "playback_drops": self._playback_drops,
            "audio_backpressure_waits": self._audio_backpressure_waits,
            "held_frame_pairs": self._held_frame_pairs,
            "video_read_failures": self._video_read_failures,
            "video_reopens": self._video_reopens,
            "runtime_s": runtime,
            "threads_total": total,
            "threads_alive": alive,
        }
        sync = self._synchronizer
        if hasattr(sync, "dropped_audio"):
            stats["sync_dropped_audio"] = sync.dropped_audio
        if hasattr(sync, "dropped_video"):
            stats["sync_dropped_video"] = sync.dropped_video
        if hasattr(sync, "max_drift_ms"):
            stats["max_drift_ms"] = sync.max_drift_ms
        if hasattr(sync, "held_frames"):
            stats["sync_held_frames"] = sync.held_frames
        sink = self._audio_sink
        if hasattr(sink, "underruns"):
            stats["playback_underruns"] = sink.underruns
        separator = self._target_separator
        if hasattr(separator, "get_status"):
            status = separator.get_status()
            stats["separator_backend"] = status.get("backend")
            stats["separator_device"] = status.get("device")
            stats["separator_precision"] = status.get("precision")
            stats["separator_healthy"] = status.get("healthy")
            stats["separator_is_real_separation"] = status.get("is_real_separation")
            stats["separator_fallback_active"] = status.get("fallback_active")
            stats["separator_failures"] = status.get("failures")
            stats["separator_avg_infer_ms"] = status.get("avg_infer_ms")
            stats["separator_peak_memory_mb"] = status.get("peak_memory_mb")
            stats["separator_load_time_ms"] = status.get("load_time_ms")
        return stats

    def start(self) -> None:
        with self._lifecycle_lock:
            if self._running:
                if self._stop_event.is_set():
                    raise RuntimeError("Pipeline cleanup is incomplete; retry stop()")
                return
            self._stop_event.clear()
            try:
                for name, device in (
                    ("microphone", self._audio_source),
                    ("playback", self._audio_sink),
                    ("camera", self._video_source),
                ):

                    self._devices_to_stop.append((name, device))
                    device.start()

                self._threads = [
                    threading.Thread(
                        target=self._audio_capture_loop,
                        name="onevoice-audio-capture",
                        daemon=True,
                    ),
                    threading.Thread(
                        target=self._video_capture_loop,
                        name="onevoice-video-capture",
                        daemon=True,
                    ),
                    threading.Thread(
                        target=self._sync_feed_loop,
                        name="onevoice-sync-feed",
                        daemon=True,
                    ),
                    threading.Thread(
                        target=self._processing_loop,
                        name="onevoice-processing",
                        daemon=True,
                    ),
                    threading.Thread(
                        target=self._playback_loop,
                        name="onevoice-playback",
                        daemon=True,
                    ),
                    threading.Thread(
                        target=self._latency_log_loop,
                        name="onevoice-latency-log",
                        daemon=True,
                    ),
                ]
                for thread in self._threads:
                    thread.start()
                with self._lock:
                    self._running = True
                    self._start_monotonic = time.monotonic()
                self._log_active_backend()
            except Exception:
                try:
                    self._stop_resources()
                except Exception:

                    logger.exception("Additional failure during startup cleanup")
                raise
            logger.info("StreamingPipeline started")

    def stop(self) -> None:

        with self._lifecycle_lock:
            self._stop_resources()

    def _stop_resources(self) -> None:
        self._stop_event.set()
        errors: list[str] = []
        first_error: Exception | None = None
        pending_devices = []

        for name, device in reversed(self._devices_to_stop):
            try:
                device.stop()
            except Exception as exc:
                logger.exception("Failed to stop %s", name)
                errors.append(f"{name}: {exc}")
                first_error = first_error or exc
                pending_devices.append((name, device))
        self._devices_to_stop = list(reversed(pending_devices))

        threads = list(self._threads)
        deadline = time.monotonic() + self._stop_timeout_s
        for thread in threads:
            if thread.ident is not None and thread is not threading.current_thread():
                thread.join(timeout=max(0.0, deadline - time.monotonic()))
        alive = [thread for thread in threads if thread.is_alive()]
        if alive:
            errors.append("workers still running: " + ", ".join(t.name for t in alive))
        with self._lock:
            if threads:
                self._threads_total = len(threads)
            self._threads_alive = len(alive)
            self._threads = alive
            self._running = bool(alive or pending_devices)
        if errors:
            raise RuntimeError("Pipeline stop incomplete: " + "; ".join(errors)) from (
                first_error
            )
        logger.info("StreamingPipeline stopped")

    def _audio_capture_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._latency.record_start("audio_capture", _now_ms())
                chunk = self._audio_source.read()
                self._latency.record_end("audio_capture", _now_ms())
                self._enqueue_audio(chunk)
            except Exception:
                if not self._stop_event.is_set():
                    logger.exception("Audio capture error")
                break

    def _enqueue_audio(self, chunk: AudioChunk) -> None:

        while not self._stop_event.is_set():
            try:
                self._audio_queue.put(chunk, timeout=0.05)
                return
            except queue.Full:
                self._audio_backpressure_waits += 1
                if self._audio_backpressure_waits % 100 == 1:
                    logger.warning(
                        "Audio capture queue full (%d waits); the processing loop "
                        "is not keeping up with real time.",
                        self._audio_backpressure_waits,
                    )
        self._audio_drops += 1

    def _video_capture_loop(self) -> None:
        consecutive_failures = 0
        while not self._stop_event.is_set():
            try:
                self._latency.record_start("video_capture", _now_ms())
                frame = self._video_source.read()
                self._latency.record_end("video_capture", _now_ms())
                consecutive_failures = 0
                self._video_queue.put(frame, timeout=0.05)
            except queue.Full:
                self._video_drops += 1
                logger.debug("Video capture queue full; dropping frame")
            except Exception:
                if self._stop_event.is_set():
                    break
                consecutive_failures += 1
                self._video_read_failures += 1
                if consecutive_failures > self._video_max_read_failures:
                    logger.exception(
                        "Video capture failed %d consecutive times; giving up",
                        consecutive_failures,
                    )
                    break
                logger.warning(
                    "Video capture read failed (%d/%d); attempting recovery",
                    consecutive_failures,
                    self._video_max_read_failures,
                )
                if not self._reopen_video_source():
                    self._stop_event.wait(self._video_retry_delay_s)

    def _reopen_video_source(self) -> bool:
        stop = getattr(self._video_source, "stop", None)
        start = getattr(self._video_source, "start", None)
        if not callable(stop) or not callable(start):
            return False
        try:
            stop()
            if self._stop_event.wait(self._video_retry_delay_s):
                return False
            start()
        except Exception:
            logger.warning("Video source reopen failed", exc_info=True)
            return False
        self._video_reopens += 1
        return True

    def _sync_feed_loop(self) -> None:
        while not self._stop_event.is_set():
            fed = False
            try:
                chunk = self._audio_queue.get(timeout=0.01)
                queue_ts = _now_ms()
                self._latency.record_start("audio_queue", chunk.timestamp_ms)
                self._latency.record_end("audio_queue", queue_ts)
                self._synchronizer.push_audio(chunk)
                fed = True
            except queue.Empty:
                pass

            try:
                frame = self._video_queue.get(timeout=0.01)
                queue_ts = _now_ms()
                self._latency.record_start("video_queue", frame.timestamp_ms)
                self._latency.record_end("video_queue", queue_ts)
                self._synchronizer.push_video(frame)
                fed = True
            except queue.Empty:
                pass

            if not fed:
                time.sleep(0.001)

    def _next_pair(self) -> tuple[AudioChunk, Frame, bool]:

        sync = self._synchronizer
        extended = getattr(sync, "get_pair_with_status", None)
        if callable(extended):
            return extended()
        audio_chunk, frame = sync.get_synchronized_pair()
        return audio_chunk, frame, False

    def _processing_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._latency.record_start("sync_wait", _now_ms())
                audio_chunk, frame, held = self._next_pair()
                self._latency.record_end("sync_wait", _now_ms())

                capture_ts = audio_chunk.timestamp_ms

                if held and self._last_tracks is not None:

                    self._held_frame_pairs += 1
                    tracks, target = self._last_tracks, self._last_target
                else:
                    self._latency.record_start("face_tracking", _now_ms())
                    tracks = self._face_tracker.process_frame(frame)
                    self._latency.record_end("face_tracking", _now_ms())

                    self._latency.record_start("target_selection", _now_ms())
                    target = self._target_selector.select_target(frame, tracks)
                    self._latency.record_end("target_selection", _now_ms())
                    self._last_tracks, self._last_target = tracks, target

                self._latency.record_start("separation", _now_ms())
                separated = self._target_separator.separate(audio_chunk, target, tracks)
                self._latency.record_end("separation", _now_ms())

                self._playback_queue.put(separated, timeout=0.05)

                playback_enqueue_ts = _now_ms()
                self._latency.record_start("end_to_end", capture_ts)
                self._latency.record_end("end_to_end", playback_enqueue_ts)

                result = PipelineResult(
                    timestamp_ms=frame.timestamp_ms,
                    separated_audio=separated,
                    processed_frame=frame,
                    active_target=target,
                    latency_metrics=self._latency.get_latencies(),
                )
                self._results_processed += 1
                if self._on_result is not None:
                    self._on_result(result)

            except TimeoutError:
                continue
            except queue.Full:
                self._playback_drops += 1
                logger.debug("Playback queue full; dropping separated chunk")
            except Exception:
                if not self._stop_event.is_set():
                    logger.exception("Processing error")
                break

    def _playback_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                chunk = self._playback_queue.get(timeout=0.05)
                self._latency.record_start("playback", _now_ms())
                self._audio_sink.write(chunk)
                self._latency.record_end("playback", _now_ms())
            except queue.Empty:
                continue
            except Exception:
                if not self._stop_event.is_set():
                    logger.exception("Playback error")
                break

    def _latency_log_loop(self) -> None:
        while not self._stop_event.wait(self._latency_log_interval_s):
            metrics = self._latency.get_latencies()
            e2e = metrics.get("end_to_end", 0.0)
            logger.info(
                "latency frames=%d e2e_avg=%.1fms capture=%.1fms sync=%.1fms "
                "track=%.1fms select=%.1fms separate=%.1fms playback=%.1fms",
                self._results_processed,
                e2e,
                metrics.get("audio_capture", 0.0),
                metrics.get("sync_wait", 0.0),
                metrics.get("face_tracking", 0.0),
                metrics.get("target_selection", 0.0),
                metrics.get("separation", 0.0),
                metrics.get("playback", 0.0),
            )

def run_mock_pipeline(duration_s: float = 5.0) -> None:

    from onevoice.audio.io import MockAudioSink, MockAudioSource
    from onevoice.selection.first_track_selector import FirstTrackSelector
    from onevoice.separation.passthrough import PassthroughSeparator
    from onevoice.video.capture import MockVideoSource
    from onevoice.video.trackers.stub_tracker import StubFaceTracker

    pipeline = StreamingPipeline(
        audio_source=MockAudioSource(),
        audio_sink=MockAudioSink(),
        video_source=MockVideoSource(),
        face_tracker=StubFaceTracker(),
        target_selector=FirstTrackSelector(),
        target_separator=PassthroughSeparator(),
    )
    logging.basicConfig(level=logging.INFO)
    pipeline.start()
    time.sleep(duration_s)
    pipeline.stop()
    print("Mock pipeline latencies:", pipeline.latency_recorder.get_latencies())

if __name__ == "__main__":
    run_mock_pipeline()
