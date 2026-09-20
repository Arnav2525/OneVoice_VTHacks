

from __future__ import annotations

import logging
import threading
import time
from typing import Any

from onevoice.core.models.frame import Frame

logger = logging.getLogger(__name__)

try:
    import cv2
except ImportError:  # pragma: no cover - optional runtime dependency
    cv2 = None  # type: ignore[assignment,misc]

def _now_ms() -> float:
    return time.monotonic() * 1000.0

class WebcamSource:

    def __init__(
        self,
        device_index: int = 0,
        width: int = 640,
        height: int = 480,
        fps: float = 30.0,
        io_timeout_ms: float = 1500.0,
        stop_wait_s: float = 1.0,
    ) -> None:
        self._device_index = device_index
        self._width = width
        self._height = height
        self._fps = fps
        self._io_timeout_ms = float(io_timeout_ms)
        self._stop_wait_s = float(stop_wait_s)
        self._capture: Any = None
        self._actual_size = (width, height)
        self._running = False
        self._lock = threading.Lock()
        self._reads_idle = threading.Condition(self._lock)
        self._reads_in_flight = 0

    def start(self) -> None:
        if cv2 is None:
            raise RuntimeError(
                "opencv-python is required for WebcamSource; "
                "pip install opencv-python-headless"
            )
        with self._lock:
            if self._running:
                return
            capture = cv2.VideoCapture(self._device_index)
            if not capture.isOpened():
                capture.release()
                raise RuntimeError(f"Failed to open webcam device {self._device_index}")
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
            capture.set(cv2.CAP_PROP_FPS, self._fps)
            getter = getattr(capture, "get", None)
            actual_w = int(getter(cv2.CAP_PROP_FRAME_WIDTH)) if getter else 0
            actual_h = int(getter(cv2.CAP_PROP_FRAME_HEIGHT)) if getter else 0
            if actual_w > 0 and actual_h > 0:
                self._actual_size = (actual_w, actual_h)
                if (actual_w, actual_h) != (self._width, self._height):
                    logger.warning(
                        "Webcam %s: asked for %dx%d but the driver gave %dx%d",
                        self._device_index,
                        self._width,
                        self._height,
                        actual_w,
                        actual_h,
                    )
                else:
                    logger.info(
                        "Webcam %s: capturing at %dx%d",
                        self._device_index,
                        actual_w,
                        actual_h,
                    )
            for name in ("CAP_PROP_OPEN_TIMEOUT_MSEC", "CAP_PROP_READ_TIMEOUT_MSEC"):
                prop = getattr(cv2, name, None)
                if prop is not None:
                    capture.set(prop, self._io_timeout_ms)
            self._capture = capture
            self._running = True

    def stop(self) -> None:
        with self._lock:
            self._running = False
            capture, self._capture = self._capture, None
            deadline = time.monotonic() + self._stop_wait_s
            while self._reads_in_flight:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    break
                self._reads_idle.wait(timeout=remaining)
        if capture is not None:
            capture.release()

    def read(self) -> Frame:
        with self._lock:
            if not self._running or self._capture is None:
                raise RuntimeError("WebcamSource is not running")
            capture = self._capture
            self._reads_in_flight += 1
        capture_start = _now_ms()
        try:
            ok, data = capture.read()
        finally:
            with self._lock:
                self._reads_in_flight -= 1
                if not self._reads_in_flight:
                    self._reads_idle.notify_all()
        if not ok:
            raise RuntimeError("Webcam frame read failed")
        return Frame(
            timestamp_ms=capture_start,
            data=data,
            metadata={
                "device_index": self._device_index,
                "width": self._actual_size[0],
                "height": self._actual_size[1],
            },
        )

class MockVideoSource:

    def __init__(self, fps: float = 30.0) -> None:
        self._fps = fps
        self._running = False
        self._frame_index = 0

    def start(self) -> None:
        self._running = True

    def stop(self) -> None:
        self._running = False

    def read(self) -> Frame:
        if not self._running:
            raise RuntimeError("MockVideoSource is not running")
        time.sleep(1.0 / self._fps)
        self._frame_index += 1
        return Frame(
            timestamp_ms=_now_ms(),
            data=None,
            metadata={
                "mock": True,
                "frame_index": self._frame_index,
                "width": 640,
                "height": 480,
            },
        )
