

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any

import yaml

from onevoice.audio.io import MockAudioSink, MockAudioSource
from onevoice.selection.first_track_selector import FirstTrackSelector
from onevoice.separation.exceptions import BackendError
from onevoice.separation.passthrough import PassthroughSeparator
from onevoice.separation.registry import create_separator
from onevoice.streaming.pipeline import StreamingPipeline
from onevoice.video.capture import MockVideoSource
from onevoice.video.trackers.stub_tracker import StubFaceTracker

logger = logging.getLogger("onevoice.streaming")

def load_experiment_config(path: str | None) -> dict[str, Any]:

    if path is None:
        return {}
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}

def _audio_params(config: dict[str, Any]) -> tuple[int, int, int]:
    audio = config.get("audio", {})
    return (
        int(audio.get("sample_rate", 16000)),
        int(audio.get("channels", 1)),
        int(audio.get("chunk_samples", 320)),
    )

def build_separator(config: dict[str, Any]) -> Any:

    backend = config.get("backend", {})
    try:
        return create_separator(backend)
    except BackendError as exc:
        logger.error(
            "could not construct backend %r (%s); using passthrough. "
            "NO REAL SEPARATION will occur.",
            backend,
            exc,
        )
        return PassthroughSeparator()

def build_face_tracker(config: dict[str, Any]) -> Any:

    video = config.get("video", {})
    name = str(video.get("tracker", "stub")).lower()
    if name in ("stub", "none"):
        return StubFaceTracker()
    if name == "iou":
        try:
            from onevoice.video.detectors import MediaPipeFaceDetector
            from onevoice.video.trackers import IouFaceTracker, TrackerConfig

            detector = MediaPipeFaceDetector(
                min_detection_confidence=float(
                    video.get("min_detection_confidence", 0.5)
                )
            )
            cfg = TrackerConfig(
                iou_threshold=float(video.get("iou_threshold", 0.3)),
                min_hits=int(video.get("min_hits", 3)),

                max_age=int(video.get("max_age", 30)),
                reid_window_ms=float(video.get("reid_window_ms", 2500.0)),
                attach_frame_bgr=bool(video.get("attach_frame_bgr", False)),
            )
            return IouFaceTracker(detector, cfg)
        except RuntimeError as exc:
            logger.warning(
                "face tracker %r unavailable (%s); falling back to StubFaceTracker. "
                "Face detection is DISABLED.",
                name,
                exc,
            )
            return StubFaceTracker()
    logger.warning("unknown video.tracker %r; using StubFaceTracker", name)
    return StubFaceTracker()

def build_target_selector(config: dict[str, Any]) -> Any:

    sel = config.get("selection", {})
    mode = str(sel.get("mode", "first_track")).lower()
    if mode in ("active_speaker", "active", "asd"):
        from onevoice.selection.active_speaker_selector import (
            ActiveSpeakerSelector,
            SelectorConfig,
        )

        cfg = SelectorConfig(
            acquire_thresh=float(sel.get("acquire_thresh", 0.55)),
            acquire_dwell_ms=float(sel.get("acquire_dwell_ms", 200.0)),
            switch_margin=float(sel.get("switch_margin", 0.15)),
            switch_dwell_ms=float(sel.get("switch_dwell_ms", 400.0)),
            coast_ms=float(sel.get("coast_ms", 500.0)),
        )
        return ActiveSpeakerSelector(cfg)
    return FirstTrackSelector()

def build_pipeline_from_config(
    config: dict[str, Any],
    live: bool = False,
    on_result: Any | None = None,
    separator: Any | None = None,
    target_selector: Any | None = None,
    record_dir: str | None = None,
) -> tuple[StreamingPipeline, float]:

    sample_rate, channels, chunk_samples = _audio_params(config)
    chunk_duration_ms = 1000.0 * chunk_samples / sample_rate

    if live:
        from onevoice.audio.io import MicrophoneSource, SpeakerSink
        from onevoice.video.capture import WebcamSource

        audio_source: Any = MicrophoneSource(sample_rate, channels, chunk_samples)
        audio_sink: Any = SpeakerSink(sample_rate, channels)
        video_source: Any = WebcamSource()
        if record_dir:
            from onevoice.audio.io import RecordingAudioSource, TeeAudioSink, WavFileSink

            record_root = Path(record_dir)
            audio_source = RecordingAudioSource(
                audio_source,
                WavFileSink(record_root / "mixture_before.wav", sample_rate, channels),
            )
            audio_sink = TeeAudioSink(
                [
                    audio_sink,
                    WavFileSink(record_root / "separated_after.wav", sample_rate, channels),
                ]
            )
    else:
        audio_source = MockAudioSource(sample_rate, channels, chunk_samples)
        audio_sink = MockAudioSink()
        video_source = MockVideoSource()

    pipeline = StreamingPipeline(
        audio_source=audio_source,
        audio_sink=audio_sink,
        video_source=video_source,
        face_tracker=build_face_tracker(config),
        target_selector=target_selector or build_target_selector(config),
        target_separator=separator or build_separator(config),
        on_result=on_result,
    )
    return pipeline, chunk_duration_ms

def _as_float(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0

def _format_status(status: dict[str, Any]) -> str:
    return (
        f"backend={status.get('backend')} device={status.get('device')} "
        f"precision={status.get('precision')} "
        f"real_separation={status.get('is_real_separation')} "
        f"fallback={status.get('fallback_active')} "
        f"avg_infer={status.get('avg_infer_ms', 0.0):.1f}ms "
        f"failures={status.get('failures')}"
    )

def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m onevoice.streaming",
        description="Run the OneVoice real-time separation pipeline from a config.",
    )
    parser.add_argument("--config", default=None, help="experiment YAML path")
    parser.add_argument(
        "--duration", type=float, default=10.0, help="run seconds (<=0 = until Ctrl-C)"
    )
    parser.add_argument(
        "--live", action="store_true", help="use real camera/microphone hardware"
    )
    parser.add_argument(
        "--record-dir",
        default=None,
        help="with --live, write mixture_before.wav + separated_after.wav here",
    )
    parser.add_argument(
        "--telemetry-interval", type=float, default=2.0, help="status print seconds"
    )
    parser.add_argument("--log-level", default="INFO", help="logging level")
    return parser

def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config = load_experiment_config(args.config)
    pipeline, chunk_ms = build_pipeline_from_config(
        config, live=args.live, record_dir=args.record_dir
    )
    logger.info("chunk duration: %.1f ms", chunk_ms)
    if args.record_dir:
        logger.info("recording mixture_before.wav + separated_after.wav to %s", args.record_dir)

    try:
        pipeline.start()
    except Exception:  # noqa: BLE001 - never let startup crash the process
        logger.exception("failed to start pipeline")
        return 1

    try:
        deadline = time.monotonic() + args.duration if args.duration > 0 else None
        interval = max(0.1, args.telemetry_interval)
        while pipeline.is_running:
            time.sleep(interval)
            status = pipeline.get_backend_status()
            logger.info("telemetry | %s", _format_status(status))
            if deadline is not None and time.monotonic() >= deadline:
                break
    except KeyboardInterrupt:
        logger.info("interrupted; shutting down")
    finally:
        pipeline.stop()
        close = getattr(pipeline.target_separator, "close", None)
        if callable(close):
            close()

    stats = pipeline.get_stats()
    logger.info(
        "final | results=%s runtime=%.1fs backend=%s real_separation=%s "
        "avg_infer=%.1fms load=%.0fms",
        stats.get("results_processed"),
        _as_float(stats.get("runtime_s", 0.0)),
        stats.get("separator_backend"),
        stats.get("separator_is_real_separation"),
        _as_float(stats.get("separator_avg_infer_ms", 0.0)),
        _as_float(stats.get("separator_load_time_ms", 0.0)),
    )
    return 0
