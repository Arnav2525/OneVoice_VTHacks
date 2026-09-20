

from __future__ import annotations

import threading
import time
from typing import Any

from onevoice.core.models.frame import Frame

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
    ) -> None:
        self._device_index = device_index
        self._width = width
        self._height = height
        self._fps = fps
        self._capture: Any = None
        self._running = False
        self._lock = threading.Lock()

    def start(self) -> None:
        if cv2 is None:
            raise RuntimeError(
                "opencv-python is required for WebcamSource; "
                "pip install opencv-python-headless"
            )
        with self._lock:
            if self._running:
                return
            self._capture = cv2.VideoCapture(self._device_index)
            if not self._capture.isOpened():
                raise RuntimeError(f"Failed to open webcam device {self._device_index}")
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
            self._capture.set(cv2.CAP_PROP_FPS, self._fps)
            self._running = True

    def stop(self) -> None:
        with self._lock:
            self._running = False
            if self._capture is not None:
                self._capture.release()
                self._capture = None

    def read(self) -> Frame:
        if not self._running or self._capture is None:
            raise RuntimeError("WebcamSource is not running")
        capture_start = _now_ms()
        ok, data = self._capture.read()
        if not ok:
            raise RuntimeError("Webcam frame read failed")
        return Frame(
            timestamp_ms=capture_start,
            data=data,
            metadata={
                "device_index": self._device_index,
                "width": self._width,
                "height": self._height,
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
