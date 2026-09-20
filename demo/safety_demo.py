

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from demo.safety import (  # noqa: E402
    DEFAULT_MONITORED_CLASSES,
    FakeClassifier,
    SafetyClassifier,
    SafetyEvent,
    SafetyGatedSink,
    SafetyMonitor,
)
from onevoice.audio.io import (  # noqa: E402
    MockAudioSink,
    MockAudioSource,
    RecordingAudioSource,
)
from onevoice.streaming.app import (  # noqa: E402
    build_face_tracker,
    build_separator,
    build_target_selector,
    load_experiment_config,
)
from onevoice.streaming.pipeline import StreamingPipeline  # noqa: E402
from onevoice.video.capture import MockVideoSource  # noqa: E402

logger = logging.getLogger("onevoice.safety_demo")

def _audio_params(config: dict[str, Any]) -> tuple[int, int, int]:
    audio = config.get("audio", {})
    return (
        int(audio.get("sample_rate", 16000)),
        int(audio.get("channels", 1)),
        int(audio.get("chunk_samples", 320)),
    )

def _print_state_change(active: bool, event: SafetyEvent | None) -> None:

    if active and event is not None:
        print(f"SAFETY OVERRIDE ACTIVE: {event.class_name} ({event.confidence:.2f})")
    else:
        print("safety override cleared")

def build_safety_pipeline(
    config: dict[str, Any],
    live: bool = False,
    classifier: SafetyClassifier | None = None,
    monitored_classes: set[str] | None = None,
    activate_thresh: float = 0.5,
    release_hold_s: float = 3.0,
    on_state_change: Callable[[bool, SafetyEvent | None], None] | None = None,
    on_result: Any | None = None,
    target_selector: Any | None = None,
) -> tuple[StreamingPipeline, SafetyMonitor]:

    sample_rate, channels, chunk_samples = _audio_params(config)

    monitor = SafetyMonitor(
        classifier or FakeClassifier(),
        monitored_classes=(
            DEFAULT_MONITORED_CLASSES
            if monitored_classes is None
            else monitored_classes
        ),
        activate_thresh=activate_thresh,
        release_hold_s=release_hold_s,
        sample_rate=sample_rate,
        on_state_change=on_state_change or _print_state_change,
    )

    if live:
        from onevoice.audio.io import MicrophoneSource, SpeakerSink
        from onevoice.video.capture import WebcamSource

        raw_source: Any = MicrophoneSource(sample_rate, channels, chunk_samples)
        playback_sink: Any = SpeakerSink(sample_rate, channels)
        video_source: Any = WebcamSource()
    else:
        raw_source = MockAudioSource(sample_rate, channels, chunk_samples)
        playback_sink = MockAudioSink()
        video_source = MockVideoSource()

    audio_source: Any = RecordingAudioSource(raw_source, monitor)

    audio_sink: Any = SafetyGatedSink(playback_sink, monitor)

    pipeline = StreamingPipeline(
        audio_source=audio_source,
        audio_sink=audio_sink,
        video_source=video_source,
        face_tracker=build_face_tracker(config),
        target_selector=target_selector or build_target_selector(config),
        target_separator=build_separator(config),
        on_result=on_result,
    )
    return pipeline, monitor

def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m demo.safety_demo",
        description="Safety/alarm passthrough demo",
    )
    parser.add_argument("--config", default=None, help="experiment YAML path")
    parser.add_argument(
        "--live", action="store_true", help="use real camera/microphone hardware"
    )
    parser.add_argument(
        "--duration", type=float, default=10.0, help="run seconds (<=0 = until Ctrl-C)"
    )
    parser.add_argument(
        "--activate-thresh",
        type=float,
        default=0.5,
        help="classifier confidence to activate",
    )
    parser.add_argument(
        "--release-hold-s",
        type=float,
        default=3.0,
        help="seconds of no event before releasing",
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
    pipeline, monitor = build_safety_pipeline(
        config,
        live=args.live,
        activate_thresh=args.activate_thresh,
        release_hold_s=args.release_hold_s,
    )
    print(
        f"OneVoice safety passthrough demo -- live={args.live} "
        f"config={args.config or '(none, defaults)'}"
    )
    print(
        "NOTE: using FakeClassifier (no real detector wired in yet) -- this "
        "run proves pipeline wiring only, it will never actually trigger."
    )

    try:

        pipeline.start()
    except Exception:  # noqa: BLE001 - never let startup crash the process
        logger.exception("failed to start pipeline")
        return 1

    try:
        deadline = time.monotonic() + args.duration if args.duration > 0 else None
        while pipeline.is_running:
            time.sleep(0.2)
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
        "final | results=%s runtime=%.1fs monitor_active=%s",
        stats.get("results_processed"),
        stats.get("runtime_s", 0.0),
        monitor.is_active(),
    )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
