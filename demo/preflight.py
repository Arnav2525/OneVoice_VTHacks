

from __future__ import annotations

import argparse
import array
import math
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

from onevoice.streaming.app import (  # noqa: E402
    build_face_tracker,
    build_pipeline_from_config,
    build_separator,
    build_target_selector,
    load_experiment_config,
)

_PASS, _WARN, _FAIL = "PASS", "WARN", "FAIL"
_SYMBOL = {
    _PASS: "\033[32m✓\033[0m",
    _WARN: "\033[33m⚠\033[0m",
    _FAIL: "\033[31m✗\033[0m",
}

class Report:

    def __init__(self) -> None:
        self._results: list[tuple[str, str, str]] = []

    def add(self, name: str, status: str, detail: str = "") -> None:
        self._results.append((name, status, detail))
        line = f"{_SYMBOL[status]} {name}"
        if detail:
            line += f" -- {detail}"
        print(line)

    @property
    def ok(self) -> bool:
        return not any(status == _FAIL for _, status, _ in self._results)

    def print_summary(self) -> None:
        n_fail = sum(1 for _, s, _ in self._results if s == _FAIL)
        n_warn = sum(1 for _, s, _ in self._results if s == _WARN)
        n_pass = sum(1 for _, s, _ in self._results if s == _PASS)
        print()
        if n_fail:
            print(
                f"\033[31m{n_fail} FAIL, {n_warn} WARN, {n_pass} PASS "
                f"-- do NOT demo until the FAILs above are fixed.\033[0m"
            )
        elif n_warn:
            print(
                f"\033[33m{n_warn} WARN, {n_pass} PASS -- no blockers, "
                f"but check the warnings above.\033[0m"
            )
        else:
            print(f"\033[32mAll {n_pass} checks passed -- ready to demo.\033[0m")

def _run(report: Report, name: str, fn: Callable[[], tuple[str, str, Any]]) -> Any:

    try:
        status, detail, payload = fn()
    except Exception as exc:  # noqa: BLE001 -- a check failing is data, not a crash
        report.add(name, _FAIL, f"{type(exc).__name__}: {exc}")
        return None
    report.add(name, status, detail)
    return payload

def _rms(samples: array.array) -> float:
    if not samples:
        return 0.0
    return math.sqrt(sum(x * x for x in samples) / len(samples))

def _check_config(config_path: str | None) -> tuple[str, str, dict[str, Any]]:
    config = load_experiment_config(config_path)
    backend_name = (config.get("backend") or {}).get("name", "passthrough (default)")
    tracker_name = (config.get("video") or {}).get("tracker", "stub (default)")
    selector_mode = (config.get("selection") or {}).get("mode", "first_track (default)")
    detail = (
        f"backend={backend_name} tracker={tracker_name} selection={selector_mode}"
    )
    return _PASS, detail, config

def _check_separator(config: dict[str, Any]) -> tuple[str, str, Any]:
    t0 = time.monotonic()
    separator = build_separator(config)

    load_error: str | None = None
    if hasattr(separator, "load"):
        try:
            separator.load()
        except Exception as exc:  # noqa: BLE001 -- a dead backend is a FAIL line, not a crash
            load_error = f"{type(exc).__name__}: {exc}"
    load_s = time.monotonic() - t0
    status = separator.get_status() if hasattr(separator, "get_status") else {}
    is_real = status.get("is_real_separation")
    backend = status.get("backend", type(separator).__name__)
    detail = f"backend={backend} load={load_s:.1f}s real_separation={is_real}"

    configured_name = (config.get("backend") or {}).get("name")
    if configured_name and configured_name != "passthrough" and is_real is False:
        cause = (
            f"the model failed to load ({load_error})"
            if load_error is not None
            else "construction fell back to passthrough (see the error logged above)"
        )
        return (
            _FAIL,
            detail + f" -- a real backend was configured but {cause}; "
            "the demo would run with NO REAL SEPARATION",
            separator,
        )
    if is_real is False:
        return (
            _WARN,
            detail + " -- passthrough backend, fine for a wiring smoke test "
            "but not real separation for the actual demo",
            separator,
        )
    return _PASS, detail, separator

def _check_face_tracker(config: dict[str, Any]) -> tuple[str, str, Any]:
    tracker = build_face_tracker(config)
    configured = str((config.get("video") or {}).get("tracker", "stub")).lower()
    actual = type(tracker).__name__
    detail = f"configured={configured} actual={actual}"
    if configured == "iou" and actual == "StubFaceTracker":
        return (
            _FAIL,
            detail + " -- requested real face tracking but fell back to the "
            "stub (see warning above); face detection is DISABLED",
            tracker,
        )
    return _PASS, detail, tracker

def _check_target_selector(config: dict[str, Any]) -> tuple[str, str, Any]:
    selector = build_target_selector(config)
    return _PASS, type(selector).__name__, selector

def _check_microphone(config: dict[str, Any], live: bool) -> tuple[str, str, None]:
    audio = config.get("audio", {})
    sample_rate = int(audio.get("sample_rate", 16000))
    channels = int(audio.get("channels", 1))
    chunk_samples = int(audio.get("chunk_samples", 320))

    if not live:
        from onevoice.audio.io import MockAudioSource

        source = MockAudioSource(sample_rate, channels, chunk_samples)
        source.start()
        chunk = source.read()
        source.stop()
        return _PASS, f"mock source OK ({len(chunk.data)} samples/chunk)", None

    from onevoice.audio.io import MicrophoneSource

    source = MicrophoneSource(sample_rate, channels, chunk_samples)
    source.start()
    try:
        n_chunks = max(1, round(sample_rate / chunk_samples))
        levels = [_rms(source.read().data) for _ in range(n_chunks)]
    finally:
        source.stop()
    peak = max(levels)
    detail = f"peak_rms={peak:.4f} over {len(levels)} chunks (~1s)"
    if peak < 0.001:
        return _WARN, detail + " -- looks silent; check the mic isn't muted", None
    return _PASS, detail, None

def _check_webcam(
    live: bool, config: dict[str, Any] | None = None
) -> tuple[str, str, None]:
    if not live:
        from onevoice.video.capture import MockVideoSource

        source = MockVideoSource()
        source.start()
        source.read()
        source.stop()
        return _PASS, "mock source OK", None

    import numpy as np

    from onevoice.video.capture import WebcamSource

    video = (config or {}).get("video", {})
    index = int(video.get("device_index", 0))
    source = WebcamSource(device_index=index)
    source.start()
    try:
        frame = source.read()
        for _ in range(9):
            frame = source.read()
    finally:
        source.stop()
    std = float(np.std(frame.data))
    detail = f"device_index={index} shape={tuple(frame.data.shape)} pixel_std={std:.1f}"
    if std < 3.0:
        hint = (
            " -- frame is flat; open the privacy shutter, check lighting, or pick "
            "another --camera-index (index 0 is often the NVIDIA Broadcast virtual "
            "camera, which is black unless NVIDIA Broadcast is running)"
        )
        return _WARN, detail + hint, None
    return _PASS, detail, None

def _check_pipeline_dry_run(
    config: dict[str, Any],
    live: bool,
    separator: Any,
    selector: Any,
    run_seconds: float,
) -> tuple[str, str, None]:
    pipeline, _chunk_ms = build_pipeline_from_config(
        config, live=live, separator=separator, target_selector=selector
    )
    pipeline.start()
    try:
        time.sleep(run_seconds)
    finally:
        pipeline.stop()
        close = getattr(pipeline.target_separator, "close", None)
        if callable(close):
            close()
    stats = pipeline.get_stats()
    processed = stats.get("results_processed", 0)
    detail = (
        f"results_processed={processed} audio_drops={stats.get('audio_drops')} "
        f"video_drops={stats.get('video_drops')} over {run_seconds:.1f}s"
    )
    if not processed:
        return (
            _FAIL,
            detail + " -- zero results produced; something in the chain is stalled",
            None,
        )
    return _PASS, detail, None

def _check_asr(model_size: str) -> tuple[str, str, None]:
    from demo.transcription import FasterWhisperTranscriber

    t0 = time.monotonic()
    FasterWhisperTranscriber(model_size=model_size)
    load_s = time.monotonic() - t0
    return _PASS, f"model_size={model_size} cold_load={load_s:.1f}s", None

def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m demo.preflight",
        description="Readiness check -- run right before demoing to judges.",
    )
    parser.add_argument("--config", default=None, help="experiment YAML path")
    parser.add_argument(
        "--live", action="store_true", help="use real camera/microphone hardware"
    )
    parser.add_argument(
        "--dry-run-seconds",
        type=float,
        default=3.0,
        help="how long to run the full pipeline for the throughput check",
    )
    parser.add_argument(
        "--check-asr",
        action="store_true",
        help="also cold-load the configured Whisper model (slow -- large-v3 "
        "takes ~230s cold; skipped by default)",
    )
    parser.add_argument(
        "--model-size",
        default="large-v3",
        help="Whisper model size to check with --check-asr",
    )
    return parser

def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)

    print(
        f"OneVoice preflight -- live={args.live} "
        f"config={args.config or '(none, defaults)'}"
    )
    print()

    report = Report()

    config = _run(report, "config", lambda: _check_config(args.config))
    if config is None:
        report.print_summary()
        return 1

    separator = _run(report, "separator", lambda: _check_separator(config))
    _run(report, "face tracker", lambda: _check_face_tracker(config))
    selector = _run(report, "target selector", lambda: _check_target_selector(config))
    _run(report, "microphone", lambda: _check_microphone(config, args.live))
    _run(report, "webcam", lambda: _check_webcam(args.live, config))

    if separator is not None and selector is not None:
        _run(
            report,
            "pipeline dry run",
            lambda: _check_pipeline_dry_run(
                config, args.live, separator, selector, args.dry_run_seconds
            ),
        )
    else:
        report.add(
            "pipeline dry run",
            _WARN,
            "skipped -- separator or target selector build failed above",
        )

    if args.check_asr:
        _run(report, "asr model load", lambda: _check_asr(args.model_size))

    report.print_summary()
    return 0 if report.ok else 1

if __name__ == "__main__":
    raise SystemExit(main())
