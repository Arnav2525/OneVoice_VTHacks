

from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from demo.level_meter import LevelMeterSink  # noqa: E402
from demo.tap_selection import (  # noqa: E402
    TrackObservingSelector,
    hit_test,
    window_click_to_native,
)
from demo.transcription import FasterWhisperTranscriber, TranscribingSink  # noqa: E402
from onevoice.audio.io import (  # noqa: E402
    MockAudioSink,
    MockAudioSource,
    RecordingAudioSource,
    TeeAudioSink,
    WavFileSink,
)
from onevoice.streaming.app import (  # noqa: E402
    build_face_tracker,
    build_separator,
    build_target_selector,
    load_experiment_config,
)
from onevoice.streaming.pipeline import StreamingPipeline  # noqa: E402
from onevoice.video.capture import MockVideoSource  # noqa: E402

logger = logging.getLogger("onevoice.transcribe_ab")

_RAW_COLOR = "\033[33m"
_ISOLATED_COLOR = "\033[32m"
_RESET = "\033[0m"

def _print_transcript(label: str, text: str, timestamp_ms: float) -> None:

    if label == "isolated":
        tag, color = "ISOLATED", _ISOLATED_COLOR
    else:
        tag, color = "RAW", _RAW_COLOR
    print(f"{color}[{tag:8s} @ {timestamp_ms:9.0f}ms]{_RESET} {text}")

class DisplayState:

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest_frame: Any = None
        self._latest_frame_size: tuple[int, int] = (640, 480)
        self._selected_id: str | None = None
        self._raw_text = ""
        self._isolated_text = ""
        self._quit = False

    def on_result(self, result: Any) -> None:

        frame = result.processed_frame
        target = result.active_target
        with self._lock:
            if frame is not None:
                self._latest_frame = frame.data
                w = frame.metadata.get("width")
                h = frame.metadata.get("height")
                if w and h:
                    self._latest_frame_size = (int(w), int(h))
            self._selected_id = (
                target.selected_speaker.track_id
                if target is not None and target.selected_speaker is not None
                else None
            )

    def on_transcript(self, label: str, text: str, timestamp_ms: float) -> None:

        with self._lock:
            if label == "isolated":
                self._isolated_text = text
            else:
                self._raw_text = text

    def snapshot(
        self,
    ) -> tuple[Any, tuple[int, int], str | None, str, str]:
        with self._lock:
            return (
                self._latest_frame,
                self._latest_frame_size,
                self._selected_id,
                self._raw_text,
                self._isolated_text,
            )

    def select(self, track_id: str, selector: TrackObservingSelector) -> None:
        print(f"[select] clicked track_id={track_id!r}")
        selector.set_manual_target(track_id)

    def clear_selection(self, selector: TrackObservingSelector) -> None:
        print("[select] cleared (back to no target)")
        selector.set_manual_target(None)

    def request_quit(self) -> None:
        self._quit = True

    @property
    def should_quit(self) -> bool:
        return self._quit

def _make_transcript_router(
    display_state: DisplayState,
) -> Callable[[str, str, float], None]:

    def _router(label: str, text: str, timestamp_ms: float) -> None:
        _print_transcript(label, text, timestamp_ms)
        display_state.on_transcript(label, text, timestamp_ms)

    return _router

_RAW_BGR = (60, 190, 235)
_ISOLATED_BGR = (90, 230, 110)
_INK = (235, 235, 235)

_CARD_MARGIN = 16
_CARD_GAP = 10
_CARD_H = 74

_FACE_UNSELECTED = (60, 60, 220)
_FACE_SELECTED = _ISOLATED_BGR

def _truncate_for_panel(text: str, max_chars: int) -> str:
    if not text:
        return "(listening...)"
    if len(text) <= max_chars:
        return text
    return "..." + text[-(max_chars - 3) :]

def _draw_transcript_panel(
    canvas: Any, top: int, tag: str, color: tuple, text: str
) -> None:

    import cv2

    from demo._render import filled_rounded_rect, label_with_background

    w = canvas.shape[1]
    left, right = _CARD_MARGIN, w - _CARD_MARGIN
    bottom = top + _CARD_H

    overlay = canvas.copy()
    filled_rounded_rect(overlay, (left, top), (right, bottom), (24, 24, 24), radius=14)
    cv2.addWeighted(overlay, 0.85, canvas, 0.15, 0, dst=canvas)
    cv2.line(canvas, (left + 16, top + 2), (right - 16, top + 2), color, 2, cv2.LINE_AA)

    label_with_background(canvas, tag, (left + 16, top + 28), color, font_scale=0.48)
    shown = _truncate_for_panel(text, max_chars=max(10, (right - left) // 11))
    cv2.putText(
        canvas,
        shown,
        (left + 16, top + 56),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        _INK,
        1,
        cv2.LINE_AA,
    )

def _draw_face_boxes(
    canvas: Any,
    tracks: list[Any],
    selected_id: str | None,
    scale_x: float,
    scale_y: float,
) -> Any:

    from demo._render import (
        apply_spotlight,
        draw_corner_brackets,
        label_with_background,
    )

    scaled = []
    selected_box: tuple[int, int, int, int] | None = None
    for track in tracks:
        x, y, w, h = track.bounding_box
        x, y, w, h = (
            int(x * scale_x),
            int(y * scale_y),
            int(w * scale_x),
            int(h * scale_y),
        )
        scaled.append((track.track_id, x, y, w, h))
        if track.track_id == selected_id:
            selected_box = (x, y, w, h)

    if selected_box is not None:
        x, y, w, h = selected_box
        canvas = apply_spotlight(canvas, x + w / 2, y + h / 2, w, h)

    for track_id, x, y, w, h in scaled:
        is_selected = track_id == selected_id
        color = _FACE_SELECTED if is_selected else _FACE_UNSELECTED
        draw_corner_brackets(canvas, x, y, w, h, color, 3 if is_selected else 2)
        label_with_background(
            canvas,
            track_id,
            (x, max(24, y - 12)),
            color,
            font_scale=0.6 if is_selected else 0.5,
        )
    return canvas

_ACCENT_DIM = (70, 140, 80)
_HEADER_H = 52

def _draw_header(canvas: Any, status_text: str, is_live: bool) -> None:

    import cv2

    from demo._render import draw_status_dot

    w = canvas.shape[1]
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (w, _HEADER_H), (16, 16, 16), -1)
    cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, dst=canvas)
    cv2.line(canvas, (0, _HEADER_H), (w, _HEADER_H), _ACCENT_DIM, 1, cv2.LINE_AA)

    cv2.putText(
        canvas,
        "ONEVOICE",
        (16, 33),
        cv2.FONT_HERSHEY_DUPLEX,
        0.7,
        _ISOLATED_BGR,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "transcript A/B",
        (128, 33),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        _ACCENT_DIM,
        1,
        cv2.LINE_AA,
    )

    from demo._render import filled_rounded_rect

    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1
    (tw, th), baseline = cv2.getTextSize(status_text, font, scale, thick)
    pad_x, pad_y = 14, 7
    pill_right = w - 16
    pill_left = pill_right - tw - 2 * pad_x - 22
    pill_top, pill_bottom = 10, 10 + th + baseline + 2 * pad_y - baseline
    filled_rounded_rect(
        canvas,
        (pill_left, pill_top),
        (pill_right, pill_bottom),
        (26, 26, 26),
        radius=14,
    )
    dot_x = pill_left + pad_x + 6
    text_x = dot_x + 16
    text_y = pill_top + (pill_bottom - pill_top) // 2 + th // 2
    draw_status_dot(
        canvas,
        (dot_x, pill_top + (pill_bottom - pill_top) // 2),
        is_live,
        _ISOLATED_BGR,
        _ACCENT_DIM,
    )
    cv2.putText(
        canvas, status_text, (text_x, text_y), font, scale, _INK, thick, cv2.LINE_AA
    )

def _draw_display_frame(
    state: DisplayState,
    window_name: str,
    selector: TrackObservingSelector,
    level_meter: LevelMeterSink,
    backend_status: dict[str, Any],
) -> Any:
    import cv2
    import numpy as np

    from demo._render import display_size, draw_vu_meter
    from demo._render import label_with_background as _label

    frame, (native_w, native_h), selected_id, raw_text, isolated_text = state.snapshot()
    disp_w, disp_h = display_size(window_name, (native_w, native_h))

    if frame is None:

        canvas = np.zeros((disp_h, disp_w, 3), dtype=np.uint8)
        cv2.putText(
            canvas,
            "MOCK VIDEO (no camera -- run with --live for real footage)",
            (24, disp_h // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (110, 110, 110),
            1,
            cv2.LINE_AA,
        )
    else:
        interp = cv2.INTER_CUBIC if disp_w > native_w else cv2.INTER_AREA
        canvas = cv2.resize(frame, (disp_w, disp_h), interpolation=interp)

    scale_x, scale_y = disp_w / native_w, disp_h / native_h
    canvas = _draw_face_boxes(
        canvas, selector.latest_tracks(), selected_id, scale_x, scale_y
    )

    real_sep = bool(backend_status.get("is_real_separation"))
    backend_name = backend_status.get("backend")
    target_line = f"target: {selected_id}" if selected_id else "click a face"
    status_text = f"{backend_name}  real_separation={real_sep}  |  {target_line}"
    _draw_header(canvas, status_text, is_live=real_sep)

    rms, peak = level_meter.snapshot()
    meter_y = _HEADER_H + 16
    meter_w = min(300, disp_w - 32)
    _label(canvas, "ISOLATED LEVEL", (16, meter_y + 14), _ISOLATED_BGR, font_scale=0.48)
    draw_vu_meter(canvas, (16, meter_y + 24), (meter_w, 14), rms, peak, _ISOLATED_BGR)

    isolated_top = disp_h - _CARD_MARGIN - _CARD_H
    raw_top = isolated_top - _CARD_GAP - _CARD_H
    _draw_transcript_panel(canvas, raw_top, "RAW", _RAW_BGR, raw_text)
    _draw_transcript_panel(
        canvas, isolated_top, "ISOLATED", _ISOLATED_BGR, isolated_text
    )

    hint_text = "click face: select/deselect    c clear all    q quit"
    hint_y = raw_top - 14
    cv2.putText(
        canvas,
        hint_text,
        (16, hint_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        hint_text,
        (16, hint_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (190, 190, 190),
        1,
        cv2.LINE_AA,
    )
    return canvas

def _check_cv2_gui_capable() -> Any | None:

    try:
        import cv2
    except ImportError:
        print(
            'ERROR: OpenCV is not installed. Run: pip install -e ".[demo]"',
            file=sys.stderr,
        )
        return None
    if not (hasattr(cv2, "imshow") and hasattr(cv2, "setMouseCallback")):
        print(
            "ERROR: this OpenCV build has no GUI support (no imshow/"
            "setMouseCallback) -- opencv-python-headless is almost certainly "
            "installed instead of (or shadowing) opencv-python. Fix:\n"
            "    pip uninstall -y opencv-python-headless\n"
            '    pip install -e ".[demo]"',
            file=sys.stderr,
        )
        return None
    return cv2

def _mouse_callback(
    event: int,
    x: int,
    y: int,
    flags: int,
    ctx: tuple[str, DisplayState, TrackObservingSelector],
) -> None:

    import cv2

    window_name, state, selector = ctx
    if event != cv2.EVENT_LBUTTONDOWN:
        return

    _, (native_w, native_h), selected_id, _, _ = state.snapshot()
    img_x, img_y = float(x), float(y)
    try:
        _, _, rect_w, rect_h = cv2.getWindowImageRect(window_name)
        img_x, img_y = window_click_to_native(x, y, rect_w, rect_h, native_w, native_h)
    except Exception:  # noqa: BLE001 - fall back to raw coords rather than crash
        pass

    tracks = selector.latest_tracks()
    track_id = hit_test(tracks, img_x, img_y)
    if track_id is None:
        return

    if track_id == selected_id:
        state.clear_selection(selector)
    else:
        state.select(track_id, selector)

def _run_display(
    pipeline: StreamingPipeline,
    state: DisplayState,
    duration: float,
    selector: TrackObservingSelector,
    level_meter: LevelMeterSink,
) -> int:
    cv2 = _check_cv2_gui_capable()
    if cv2 is None:
        return 1

    window_name = "OneVoice - transcript A-B"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 720)
    cv2.setMouseCallback(
        window_name, _mouse_callback, (window_name, state, selector)
    )  # type: ignore[arg-type]

    deadline = time.monotonic() + duration if duration > 0 else None
    try:
        while pipeline.is_running and not state.should_quit:
            canvas = _draw_display_frame(
                state, window_name, selector, level_meter, pipeline.get_backend_status()
            )
            cv2.imshow(window_name, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                state.request_quit()
            elif key == ord("c"):
                state.clear_selection(selector)
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                state.request_quit()
            if deadline is not None and time.monotonic() >= deadline:
                break
    finally:
        cv2.destroyAllWindows()
    return 0

def _audio_params(config: dict[str, Any]) -> tuple[int, int, int]:
    audio = config.get("audio", {})
    return (
        int(audio.get("sample_rate", 16000)),
        int(audio.get("channels", 1)),
        int(audio.get("chunk_samples", 320)),
    )

def build_ab_pipeline(
    config: dict[str, Any],
    live: bool = False,
    save_dir: str | None = None,
    save_raw_too: bool = False,
    mixture_transcriber: Any | None = None,
    isolated_transcriber: Any | None = None,
    max_segment_s: float = 6.0,
    enable_vad: bool = True,
    silence_hold_s: float = 0.6,
    min_segment_s: float = 0.0,
    on_transcript: Callable[[str, str, float], None] | None = None,
    on_result: Any | None = None,
    target_selector: Any | None = None,
    level_meter: Any | None = None,
    model_size: str = "large-v3",
) -> tuple[StreamingPipeline, TranscribingSink, TranscribingSink]:

    sample_rate, channels, chunk_samples = _audio_params(config)
    silence_thresh = 0.02 if enable_vad else None
    transcript_callback = on_transcript or _print_transcript

    mixture_sink = TranscribingSink(
        "raw",
        mixture_transcriber or FasterWhisperTranscriber(model_size=model_size),
        transcript_callback,
        sample_rate=sample_rate,
        max_segment_s=max_segment_s,
        silence_thresh=silence_thresh,
        silence_hold_s=silence_hold_s,
        min_segment_s=min_segment_s,
    )
    isolated_sink = TranscribingSink(
        "isolated",
        isolated_transcriber or FasterWhisperTranscriber(model_size=model_size),
        transcript_callback,
        sample_rate=sample_rate,
        max_segment_s=max_segment_s,
        silence_thresh=silence_thresh,
        silence_hold_s=silence_hold_s,
        min_segment_s=min_segment_s,
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

    mixture_recorder: Any = mixture_sink
    if save_dir and save_raw_too:
        save_root = Path(save_dir)
        mixture_recorder = TeeAudioSink(
            [
                mixture_sink,
                WavFileSink(save_root / "mixture.wav", sample_rate, channels),
            ]
        )
    audio_source: Any = RecordingAudioSource(raw_source, mixture_recorder)

    isolated_sinks: list[Any] = [playback_sink, isolated_sink]
    if level_meter is not None:
        isolated_sinks.append(level_meter)
    if save_dir:
        save_root = Path(save_dir)
        isolated_sinks.append(
            WavFileSink(save_root / "isolated.wav", sample_rate, channels)
        )
    audio_sink: Any = TeeAudioSink(isolated_sinks)

    pipeline = StreamingPipeline(
        audio_source=audio_source,
        audio_sink=audio_sink,
        video_source=video_source,
        face_tracker=build_face_tracker(config),
        target_selector=target_selector or build_target_selector(config),
        target_separator=build_separator(config),
        on_result=on_result,
    )
    return pipeline, mixture_sink, isolated_sink

def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m demo.transcribe_ab",
        description="Raw-vs-isolated live transcription A/B demo",
    )
    parser.add_argument("--config", default=None, help="experiment YAML path")
    parser.add_argument(
        "--live", action="store_true", help="use real camera/microphone hardware"
    )
    parser.add_argument(
        "--duration", type=float, default=30.0, help="run seconds (<=0 = until Ctrl-C)"
    )
    parser.add_argument(
        "--model-size",
        default="large-v3",
        help="faster-whisper model size for both transcribers (default: large-v3). "
        "Pass a smaller size (e.g. small, medium) for faster but less accurate transcription.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="persist the ISOLATED stream's audio to <save-dir>/isolated.wav "
        "(the raw mixture is never saved without --save-raw-too on top of this)",
    )
    parser.add_argument(
        "--save-raw-too",
        action="store_true",
        help="with --save, ALSO persist the raw mixture to <save-dir>/mixture.wav "
        "-- an explicit extra opt-in, since this defeats the privacy pitch",
    )
    parser.add_argument(
        "--save-dir",
        default="data/transcribe_ab_out",
        help="directory for --save output",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="open a cv2 window: video feed with clickable face boxes, RAW/ISOLATED "
        "transcript panels, and a live VU meter for the isolated stream (needs a "
        'GUI-capable OpenCV build: pip install -e ".[demo]")',
    )
    parser.add_argument("--log-level", default="WARNING")
    return parser

def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.WARNING),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config = load_experiment_config(args.config)
    if not args.live:

        config = dict(config)
        config["video"] = dict(config.get("video", {}))
        config["video"]["tracker"] = "stub"
        print("Mock mode (no --live): using StubFaceTracker, one fake centered box.")
    save_dir = args.save_dir if args.save else None
    save_raw_too = args.save and args.save_raw_too

    print(
        f"OneVoice transcription A/B -- live={args.live} config={args.config} "
        f"save={'ON (' + save_dir + ')' if save_dir else 'off'} "
        f"save_raw_too={save_raw_too}"
    )
    print(
        f"{_RAW_COLOR}[RAW]{_RESET} = whatever the microphone actually picked up "
        f"(everyone talking at once). {_ISOLATED_COLOR}[ISOLATED]{_RESET} = only "
        "the currently-selected face's voice. Watch them diverge.\n"
    )

    display_state = DisplayState() if args.display else None
    on_result = display_state.on_result if display_state is not None else None
    on_transcript = (
        _make_transcript_router(display_state)
        if display_state is not None
        else _print_transcript
    )
    selector = (
        TrackObservingSelector(build_target_selector(config))
        if display_state is not None
        else None
    )
    level_meter = LevelMeterSink() if display_state is not None else None

    pipeline, mixture_sink, isolated_sink = build_ab_pipeline(
        config,
        live=args.live,
        save_dir=save_dir,
        save_raw_too=save_raw_too,
        on_transcript=on_transcript,
        on_result=on_result,
        target_selector=selector,
        level_meter=level_meter,
        model_size=args.model_size,
    )

    try:
        pipeline.start()
    except Exception:  # noqa: BLE001 - never let startup crash the process
        logger.exception("failed to start pipeline")
        return 1

    exit_code = 0
    try:
        if display_state is not None:

            exit_code = _run_display(
                pipeline, display_state, args.duration, selector, level_meter
            )
        else:
            deadline = time.monotonic() + args.duration if args.duration > 0 else None
            while pipeline.is_running:
                time.sleep(0.2)
                if deadline is not None and time.monotonic() >= deadline:
                    break
    except KeyboardInterrupt:
        print("\ninterrupted; shutting down")
    finally:
        pipeline.stop()
        close = getattr(pipeline.target_separator, "close", None)
        if callable(close):
            close()

    return exit_code

if __name__ == "__main__":
    raise SystemExit(main())
