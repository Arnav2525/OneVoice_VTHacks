

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

import cv2

from demo.devices import format_device_list, list_audio_devices
from demo.session_runtime import SessionRunner
from demo.session_view import face_at, inside, render_session
from onevoice.streaming.app import load_experiment_config

class SessionWindow:

    def __init__(self, runner: SessionRunner, *, windowed: bool = False) -> None:
        self.runner = runner
        self.name = "OneVoice - listening session"
        self.fullscreen = not windowed
        self.quitting = False
        self.canvas_size = (1280, 800)
        self.controls: dict[str, tuple[int, int, int, int]] = {}
        self.video_rect = (0, 0, 1, 1)
        self.native_size = (640, 480)
        self.drawn_tracks: list[Any] = []
        self._last_status: tuple[str, str] | None = None
        self._last_record: tuple[Any, ...] | None = None

    def apply_window_mode(self) -> None:
        if sys.platform == "win32":
            if self.fullscreen:
                import ctypes

                user32 = ctypes.windll.user32
                cv2.moveWindow(self.name, 0, 0)
                cv2.resizeWindow(
                    self.name,
                    user32.GetSystemMetrics(0),
                    max(640, user32.GetSystemMetrics(1) - 80),
                )
            else:
                cv2.resizeWindow(self.name, 1280, 800)
        else:
            cv2.setWindowProperty(
                self.name,
                cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN if self.fullscreen else cv2.WINDOW_NORMAL,
            )
            if not self.fullscreen:
                cv2.resizeWindow(self.name, 1280, 800)

    def action(self, name: str) -> None:
        if self.quitting:
            return
        if name == "session":
            if self.runner.busy:
                self.runner.stop()
            else:
                self.runner.start()
        elif name == "record":
            self.runner.toggle_recording()
        elif name == "clear":
            self.runner.state.select(None)

    def click(self, x: float, y: float) -> None:
        for name, rect in self.controls.items():
            if inside(x, y, rect):
                self.action(name)
                return
        selected = face_at(
            x,
            y,
            self.video_rect,
            self.native_size,
            self.drawn_tracks,
        )
        if selected is not None and not self.quitting:
            self.runner.state.select(selected)

    def render(self) -> Any:
        try:
            _, _, width, height = cv2.getWindowImageRect(self.name)
            size = (width, height) if width > 0 and height > 0 else self.canvas_size
        except cv2.error:
            size = self.canvas_size
        view = self.runner.state.snapshot()
        recording = self.runner.recorder.snapshot()
        self.drawn_tracks = self.runner.selector.latest_tracks()
        canvas, self.controls, self.video_rect, self.native_size = render_session(
            view,
            self.runner.frame(),
            self.drawn_tracks,
            self.runner.state.person_name,
            recording,
            size=size,
            synthetic=not self.runner.live,
            record_busy=self.runner.record_busy,
        )
        self.canvas_size = (canvas.shape[1], canvas.shape[0])
        status = (view.phase, view.detail)
        if status != self._last_status:
            print(f"[{view.phase}] {view.title} - {view.detail}", flush=True)
            self._last_status = status
        record_status = (recording.active, recording.path, recording.error)
        if record_status != self._last_record:
            if recording.error:
                print(f"[recording error] {recording.error}", file=sys.stderr)
            elif recording.path:
                verb = "Recording to" if recording.active else "Saved clip to"
                print(f"{verb} {recording.path}", flush=True)
            self._last_record = record_status
        return canvas

def _mouse_callback(
    event: int,
    x: int,
    y: int,
    flags: int,
    app: Any,
) -> None:
    if event == cv2.EVENT_LBUTTONDOWN:

        app.click(float(x), float(y))

def run(
    config_path: str,
    live: bool,
    record_dir: str | None = None,
    *,
    preview: bool = False,
    windowed: bool = False,
    native: bool = False,
    open_browser: bool = True,
    port: int = 0,
    camera_index: int | None = None,
    input_device: int | str | None = None,
    output_device: int | str | None = None,
) -> int:
    config = load_experiment_config(config_path)

    if camera_index is not None:
        if camera_index < 0:
            raise ValueError("Camera index must be zero or greater")
        config.setdefault("video", {})["device_index"] = camera_index
    if input_device is not None:
        config.setdefault("audio", {})["input_device"] = input_device
    if output_device is not None:
        config.setdefault("audio", {})["output_device"] = output_device
    runner = SessionRunner(
        config,
        live=live,
        preview=preview,
        record_root=Path(record_dir) if record_dir else REPO_ROOT / "runs/sessions",
    )
    if not native:
        from demo.web_session import run_web

        return run_web(runner, open_browser=open_browser, port=port)
    app = SessionWindow(runner, windowed=windowed)
    clean = True
    try:
        cv2.namedWindow(app.name, cv2.WINDOW_NORMAL)
        app.apply_window_mode()
        cv2.setMouseCallback(app.name, _mouse_callback, app)
        while True:
            cv2.imshow(app.name, app.render())
            key = cv2.waitKey(20) & 0xFF
            if key in (ord("q"), 27):
                app.quitting = True
                runner.stop()
            elif key == ord(" "):
                app.action("session")
            elif key == ord("r"):
                app.action("record")
            elif key == ord("c"):
                app.action("clear")
            elif key == ord("f"):
                app.fullscreen = not app.fullscreen
                app.apply_window_mode()
            if cv2.getWindowProperty(app.name, cv2.WND_PROP_VISIBLE) < 1:
                app.quitting = True
                runner.stop()
            if app.quitting and not runner.busy:
                break
    except cv2.error as exc:
        print(f"OpenCV window error: {exc}", file=sys.stderr)
        print(
            'Use a GUI-capable OpenCV build: pip install -e ".[demo]"', file=sys.stderr
        )
        return 1
    finally:
        clean = runner.close()
        cv2.destroyAllWindows()
        if not clean:
            print(
                "Session cleanup is incomplete; see the session error above.",
                file=sys.stderr,
            )
    return 0 if clean else 1

def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "configs/experiments/dolphin_tap.yaml"),
        help="experiment YAML",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="enable real camera/microphone after Start listening",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="CPU capture preview; isolation and playback disabled",
    )
    parser.add_argument(
        "--record-dir",
        default=None,
        help="clip destination; saves only after Record clip",
    )
    parser.add_argument(
        "--windowed", action="store_true", help="start the --native UI in a window"
    )
    parser.add_argument(
        "--native", action="store_true", help="use the legacy OpenCV UI"
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="print the local UI URL without opening it",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=0,
        help="local browser port (default: choose an available port)",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=None,
        help="camera index for this run (external camera is often 1)",
    )
    parser.add_argument(
        "--input-device",
        default=None,
        help="microphone name or numeric ID (e.g. C270); 'default' uses the OS default",
    )
    parser.add_argument(
        "--output-device",
        default=None,
        help="headphone name or listed numeric ID; default uses OS default",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="list microphones and playback devices without opening capture or the UI",
    )
    return parser

def main(argv: list[str] | None = None) -> int:
    parser = _build_argparser()
    args = parser.parse_args(argv)
    if args.list_devices:
        try:
            print(format_device_list(list_audio_devices()))
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        return 0
    if args.camera_index is not None and args.camera_index < 0:
        parser.error("--camera-index must be zero or greater")
    return run(
        args.config,
        args.live,
        args.record_dir,
        preview=args.preview,
        windowed=args.windowed,
        native=args.native,
        open_browser=not args.no_browser,
        port=args.port,
        camera_index=args.camera_index,
        input_device=args.input_device,
        output_device=args.output_device,
    )

if __name__ == "__main__":
    raise SystemExit(main())
