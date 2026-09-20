from __future__ import annotations

import copy
import logging
import math
import threading
from dataclasses import replace
from pathlib import Path
from typing import Any

from demo.devices import resolve_audio_device
from demo.session_captions import SessionCaptions, build_transcriber_factory
from demo.session_recording import SessionRecorder
from demo.session_state import SessionState
from demo.tap_selection import TrackObservingSelector
from onevoice.audio.cleanup import RumbleFilter
from onevoice.audio.io import MockAudioSink, MockAudioSource
from onevoice.core.models.speaker_track import SpeakerTrack
from onevoice.core.models.target_selection import TargetSelection
from onevoice.streaming.app import build_face_tracker, build_separator
from onevoice.streaming.pipeline import StreamingPipeline
from onevoice.video.capture import MockVideoSource

logger = logging.getLogger(__name__)


class PreviewTracker:
    def __init__(self) -> None:
        self.hidden: set[str] = set()

    def process_frame(self, frame: Any) -> list[SpeakerTrack]:
        return [
            SpeakerTrack(
                track_id=f"preview-{i}",
                bounding_box=(45 + i * 200, 120, 150, 210),
                confidence=1.0,
                metadata={"visible": True, "synthetic": True},
            )
            for i in range(3)
            if f"preview-{i}" not in self.hidden
        ]


class SessionSelector:
    def __init__(self, state: SessionState) -> None:
        self.state = state

    def select_target(self, frame: Any, tracks: list[SpeakerTrack]) -> TargetSelection:
        self.state.observe_tracks(tracks)
        requested, _ = self.state.selection()
        track = next(
            (
                t
                for t in tracks
                if t.track_id == requested and t.metadata.get("visible", True)
            ),
            None,
        )
        return TargetSelection(frame.timestamp_ms, track)


class ObservedSource:
    def __init__(self, source: Any, callback: Any) -> None:
        self.source, self.callback = source, callback

    def start(self) -> None:
        self.source.start()

    def stop(self) -> None:
        self.source.stop()

    def read(self) -> Any:
        value = self.source.read()
        self.callback(value)
        return value


class EpochSeparator:
    def __init__(self, inner: Any, state: SessionState) -> None:
        self.inner, self.state = inner, state

    def separate(self, audio: Any, target: Any, tracks: Any) -> Any:
        requested, epoch = self.state.selection()
        actual = target.selected_speaker
        matches = actual is not None and actual.track_id == requested
        if actual is not None:
            target = replace(
                target,
                selected_speaker=replace(
                    actual,
                    metadata={**actual.metadata, "ui_selection_epoch": epoch},
                ),
            )
        result = self.inner.separate(audio, target, tracks)
        return replace(
            result,
            metadata={
                **result.metadata,
                "ui_selection_epoch": epoch if matches else -1,
            },
        )

    def get_status(self) -> dict[str, Any]:
        return dict(self.inner.get_status())

    def close(self) -> None:
        close = getattr(self.inner, "close", None)
        if callable(close):
            close()


def _ramp(samples: Any, *, rising: bool, length: int) -> list[float]:

    values = [float(v) for v in samples]
    span = min(length, len(values))
    if span <= 0:
        return values
    for i in range(span):
        phase = (i + 0.5) / span
        gain = 0.5 - 0.5 * math.cos(math.pi * phase)
        values[i] *= gain if rising else (1.0 - gain)
    if not rising:
        for i in range(span, len(values)):
            values[i] = 0.0
    return values


class SessionSink:
    def __init__(
        self,
        sink: Any,
        state: SessionState,
        separator: Any,
        recorder: SessionRecorder,
        captions: SessionCaptions,
        cleanup: RumbleFilter | None = None,
        denoiser: Any = None,
    ) -> None:
        self.sink, self.state = sink, state
        self.separator, self.recorder, self.captions = separator, recorder, captions
        self.cleanup = cleanup
        self.denoiser = denoiser
        self._was_allowed = False
        self._fade_samples = 128

    def start(self) -> None:
        self._was_allowed = False
        if self.cleanup is not None:
            self.cleanup.reset()
        if self.denoiser is not None:
            self.denoiser.reset()
        self.sink.start()

    def stop(self) -> None:
        self.sink.stop()

    def write(self, chunk: Any) -> None:
        status = self.separator.get_status()
        allowed = (
            self.state.output_allowed(chunk)
            and status.get("is_real_separation") is True
            and not status.get("fallback_active")
        )
        for processor in (self.cleanup, self.denoiser):
            if processor is None:
                continue
            if allowed:
                chunk = processor.process(chunk)

                allowed = self.state.output_allowed(chunk)
            if not allowed:
                processor.reset()
        if allowed and not self._was_allowed:
            delivered = replace(
                chunk,
                data=_ramp(chunk.data, rising=True, length=self._fade_samples),
                metadata={**chunk.metadata, "mute_ramp": "in"},
            )
        elif allowed:
            delivered = chunk
        elif self._was_allowed:
            delivered = replace(
                chunk,
                data=_ramp(chunk.data, rising=False, length=self._fade_samples),
                metadata={
                    **chunk.metadata,
                    "ui_muted": True,
                    "output_ready": False,
                    "mute_ramp": "out",
                },
            )
        else:
            delivered = replace(
                chunk,
                data=[0.0] * len(chunk.data),
                metadata={**chunk.metadata, "ui_muted": True, "output_ready": False},
            )
        self._was_allowed = allowed
        self.sink.write(delivered)
        self.state.observe_output(delivered, status)
        self.recorder.on_output(delivered)

        self.captions.on_output(chunk, allowed)


class SessionRunner:
    CAMERA_SOURCES: dict[str, int] = {"built_in": 0, "external": 1}

    def __init__(
        self,
        config: dict[str, Any],
        *,
        live: bool = False,
        preview: bool = False,
        record_root: Path = Path("runs/sessions"),
        factory: Any = None,
    ) -> None:
        self.config = copy.deepcopy(config)
        self.live = live
        self.preview = (
            preview
            or not live
            or config.get("backend", {}).get("name") == "passthrough"
        )
        self.state = SessionState("preview" if self.preview else "live")
        self.recorder = SessionRecorder()
        self.captions = SessionCaptions(
            self.state, build_transcriber_factory(self.config)
        )
        self.record_root = record_root
        self._record_control_lock = threading.Lock()
        self._record_io_lock = threading.Lock()
        self._record_thread: threading.Thread | None = None
        self.selector = TrackObservingSelector(SessionSelector(self.state))
        self.preview_tracker = PreviewTracker()
        self._thread: threading.Thread | None = None
        self._cancel = threading.Event()
        self._frame_lock = threading.Lock()
        self._frame: Any = None
        self._factory = factory or self._build
        self._lifecycle_lock = threading.Lock()

        self._pipeline: Any = None
        self._separator: Any = None
        self._identity_tracker: Any = None
        self._recording_cleanup_pending = False
        self._camera_override: int | None = None

    @property
    def camera_source(self) -> str:
        index = self._camera_override
        if index is None:
            index = int(self.config.get("video", {}).get("device_index", 0))
        return "built_in" if index == 0 else "external"

    @property
    def busy(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def set_camera_source(self, source: str) -> bool:
        if source not in self.CAMERA_SOURCES:
            return False
        if self.busy or self.state.snapshot().active:
            return False
        self._camera_override = self.CAMERA_SOURCES[source]
        return True

    def start(self) -> None:

        with self._lifecycle_lock:
            if self.busy:
                return
            if self._cleanup_pending():
                self.state.begin_stop()
            else:
                self.state.begin_start()
            self._cancel.clear()
            self._launch_worker(start_session=True)

    def stop(self) -> None:
        with self._lifecycle_lock:
            if self.busy or self._cleanup_pending():
                self.state.begin_stop()
                self._cancel.set()
                if not self.busy:
                    self._launch_worker(start_session=False)

    def close(self, timeout: float = 5.0) -> bool:

        self.stop()
        with self._lifecycle_lock:
            thread = self._thread
        if thread is not None and thread.ident is not None:
            thread.join(timeout=timeout)
        self.captions.close()
        return not self.busy and not self._cleanup_pending()

    def _cleanup_pending(self) -> bool:
        return (
            self._pipeline is not None
            or self._separator is not None
            or self._identity_tracker is not None
            or self._recording_cleanup_pending
        )

    def _launch_worker(self, *, start_session: bool) -> None:
        self._thread = threading.Thread(
            target=self._run,
            args=(start_session,),
            daemon=True,
            name="onevoice-session",
        )
        try:
            self._thread.start()
        except Exception as exc:
            self._thread = None
            self.state.mark_stopped(str(exc) or type(exc).__name__)

    def frame(self) -> Any:
        with self._frame_lock:
            return self._frame

    def identity_status(self) -> dict[str, Any]:
        tracker = self._identity_tracker
        return tracker.status() if tracker is not None else {}

    def _on_frame(self, frame: Any) -> None:
        self.state.observe_frame(frame)
        with self._frame_lock:
            self._frame = frame

    def _on_input(self, chunk: Any) -> None:
        chunk = replace(
            chunk,
            metadata={
                **chunk.metadata,
                "input_device": getattr(self, "_input_device", None),
                "input_device_requested": str(
                    self.config.get("audio", {}).get("input_device", "default")
                ),
            },
        )
        self.state.observe_input(chunk)
        self.recorder.on_input(chunk)

    @property
    def record_busy(self) -> bool:

        with self._record_control_lock:
            return bool(self._record_thread and self._record_thread.is_alive())

    def _recording_allowed(self) -> bool:
        return not self._cancel.is_set() and self.state.snapshot().phase in (
            "listening",
            "focusing",
            "isolating",
            "target_lost",
        )

    def toggle_captions(self) -> None:
        self.captions.toggle()

    def toggle_recording(self) -> None:

        with self._record_control_lock:
            if self._record_thread and self._record_thread.is_alive():
                return
            stopping = self.recorder.snapshot().active
            if not stopping and not self._recording_allowed():
                return
            self._record_thread = threading.Thread(
                target=self._change_recording,
                args=(stopping,),
                name="onevoice-record-control",
                daemon=True,
            )
            self._record_thread.start()

    def _change_recording(self, stopping: bool) -> None:
        try:
            with self._record_io_lock:
                if stopping:
                    self.recorder.stop_clip()
                elif self._recording_allowed():
                    self.recorder.start_clip(self.record_root, self.state.mode)
                    if not self._recording_allowed():
                        self.recorder.stop_clip()
        except Exception:
            logger.exception("Could not change clip recording")

    def _finish_recording(self) -> None:

        with self._record_control_lock:
            worker = self._record_thread
        if worker is not None and worker.ident is not None:
            worker.join()
        with self._record_io_lock:
            self.recorder.stop_clip()

    def _build(self) -> tuple[Any, Any]:
        config = copy.deepcopy(self.config)
        if self.preview:
            config["backend"] = {"name": "passthrough"}
        elif config.get("backend", {}).get("name") == "dolphin":
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError(
                    "Live isolation needs a GPU. Use --preview on this laptop."
                )
        audio = config.get("audio", {})
        rate, channels, chunk = (
            int(audio.get("sample_rate", 16000)),
            int(audio.get("channels", 1)),
            int(audio.get("chunk_samples", 320)),
        )
        if self.live:
            from demo.face_identity import LocalFaceEncoder, SessionFaceTracker
            from onevoice.audio.io import MicrophoneSource, SpeakerSink
            from onevoice.video.capture import WebcamSource
            from onevoice.video.trackers.stub_tracker import StubFaceTracker

            identity_enabled = bool(
                config.get("video", {}).get("identity_matching", True)
            )
            tracker = build_face_tracker(config)
            if isinstance(tracker, StubFaceTracker):
                raise RuntimeError(
                    "Face detection is unavailable. Check the perception installation."
                )
            if identity_enabled:
                self.state.starting_detail("Preparing local face matching...")
                encoder = LocalFaceEncoder()
                tracker = SessionFaceTracker(tracker, encoder)
                self._identity_tracker = tracker
            else:
                logger.info(
                    "Identity matching disabled (video.identity_matching: false); "
                    "using geometric track IDs."
                )
            self._input_device = resolve_audio_device(
                audio.get("input_device"),
                kind="input",
                sample_rate=rate,
                channels=channels,
            )
            output_device = (
                None
                if self.preview
                else resolve_audio_device(
                    audio.get("output_device"),
                    kind="output",
                    sample_rate=rate,
                    channels=channels,
                )
            )
            raw = MicrophoneSource(rate, channels, chunk, device=self._input_device)

            sink = (
                MockAudioSink()
                if self.preview
                else SpeakerSink(
                    rate,
                    channels,
                    device=output_device,
                )
            )
            video_config = config.get("video", {})
            camera_device_index = (
                self._camera_override
                if self._camera_override is not None
                else int(video_config.get("device_index", 0))
            )
            video = WebcamSource(
                device_index=camera_device_index,
                width=int(video_config.get("width", 640)),
                height=int(video_config.get("height", 480)),
                fps=float(video_config.get("fps", 30.0)),
            )
            logger.info(
                "Capture: camera %s (%s), microphone %s; playback %s",
                camera_device_index,
                self.camera_source,
                self._input_device,
                output_device,
            )
        else:
            tracker = self.preview_tracker
            raw, sink, video = (
                MockAudioSource(rate, channels, chunk),
                MockAudioSink(),
                MockVideoSource(),
            )
        cleanup_config = audio.get("cleanup", {})
        cleanup = (
            RumbleFilter(float(cleanup_config.get("cutoff_hz", 80.0)))
            if cleanup_config.get("enabled", False)
            else None
        )
        denoiser = None
        denoise_config = audio.get("denoise", {})
        if denoise_config.get("enabled", False) and not self.preview:
            from onevoice.audio.denoise import GtcrnDenoiser

            model_path = Path(
                denoise_config.get(
                    "model_path",
                    "checkpoints/gtcrn/gtcrn_simple.onnx",
                )
            )
            if not model_path.is_absolute():
                model_path = Path(__file__).resolve().parents[1] / model_path
            denoiser = GtcrnDenoiser(model_path=model_path)
        separator = EpochSeparator(build_separator(config), self.state)
        pipeline = StreamingPipeline(
            audio_source=ObservedSource(raw, self._on_input),
            audio_sink=SessionSink(
                sink,
                self.state,
                separator,
                self.recorder,
                self.captions,
                cleanup,
                denoiser,
            ),
            video_source=ObservedSource(video, self._on_frame),
            face_tracker=tracker,
            target_selector=self.selector,
            target_separator=separator,
        )
        return pipeline, separator

    def _cleanup_resources(self) -> str | None:
        self.state.begin_stop()

        self.captions.retire_current_sink()
        errors = []
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
                if getattr(self._pipeline, "is_running", False):
                    raise RuntimeError("Pipeline workers or devices have not stopped")
            except Exception as exc:
                errors.append(str(exc) or type(exc).__name__)
            else:
                self._pipeline = None

        if self._pipeline is None and self._identity_tracker is not None:
            try:
                self._identity_tracker.close()
            except Exception as exc:
                errors.append(str(exc) or type(exc).__name__)
            else:
                self._identity_tracker = None
        if self._pipeline is None and self._separator is not None:
            try:
                self._separator.close()
            except Exception as exc:
                errors.append(str(exc) or type(exc).__name__)
            else:
                self._separator = None
        try:
            self._finish_recording()
        except Exception as exc:
            self._recording_cleanup_pending = True
            errors.append(str(exc) or type(exc).__name__)
        else:
            self._recording_cleanup_pending = False
        return "; ".join(errors) or None

    def _finish_session(self, error: str | None) -> None:
        with self._frame_lock:
            self._frame = None
        self.state.mark_stopped(error)

    def _run(self, start_session: bool = True) -> None:
        if self._cleanup_pending():
            error = self._cleanup_resources()
            if error:
                self._finish_session(error)
                return
        with self._lifecycle_lock:
            if not start_session or self._cancel.is_set():
                self._finish_session(None)
                return
            self.state.begin_start()
            self.selector = TrackObservingSelector(SessionSelector(self.state))
            self.preview_tracker = PreviewTracker()
            with self._frame_lock:
                self._frame = None

        error = None
        try:
            self._pipeline, self._separator = self._factory()
            inner = getattr(self._separator, "inner", self._separator)
            load = getattr(inner, "load", None)
            if callable(load) and not self._cancel.is_set():
                self.state.starting_detail("Preparing voice isolation...")
                load()
            if not self._cancel.is_set():
                self._pipeline.start()
                self.state.mark_running()
                self._cancel.wait()
        except Exception as exc:
            error = str(exc) or type(exc).__name__
        finally:
            cleanup_error = self._cleanup_resources()
            if cleanup_error:
                error = f"{error + '; ' if error else ''}{cleanup_error}"
            self._finish_session(error)
