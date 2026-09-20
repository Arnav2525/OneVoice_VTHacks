import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from onevoice.video import capture


def _fake_cv2(webcam):
    return SimpleNamespace(
        VideoCapture=lambda _index: webcam,
        CAP_PROP_FRAME_WIDTH=3,
        CAP_PROP_FRAME_HEIGHT=4,
        CAP_PROP_FPS=5,
        CAP_PROP_OPEN_TIMEOUT_MSEC=53,
        CAP_PROP_READ_TIMEOUT_MSEC=54,
    )


def test_stop_waits_for_in_flight_read_before_release(monkeypatch):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    read_entered = threading.Event()
    release_allowed = threading.Event()
    events = []

    def slow_read():
        read_entered.set()
        release_allowed.wait(2.0)
        events.append("read_done")
        return True, frame

    webcam = SimpleNamespace(
        isOpened=lambda: True,
        set=lambda *_: None,
        read=slow_read,
        release=lambda: events.append("release"),
    )
    monkeypatch.setattr(capture, "cv2", _fake_cv2(webcam))

    source = capture.WebcamSource(stop_wait_s=2.0)
    source.start()

    reader = threading.Thread(target=source.read, daemon=True)
    reader.start()
    assert read_entered.wait(2.0)

    stopper = threading.Thread(target=source.stop, daemon=True)
    stopper.start()
    time.sleep(0.2)
    assert events == []

    release_allowed.set()
    reader.join(2.0)
    stopper.join(2.0)
    assert events == ["read_done", "release"]


def test_stop_releases_after_deadline_when_read_hangs(monkeypatch):
    read_entered = threading.Event()
    never = threading.Event()
    events = []

    def hanging_read():
        read_entered.set()
        never.wait(5.0)
        return True, np.zeros((480, 640, 3), dtype=np.uint8)

    webcam = SimpleNamespace(
        isOpened=lambda: True,
        set=lambda *_: None,
        read=hanging_read,
        release=lambda: events.append("release"),
    )
    monkeypatch.setattr(capture, "cv2", _fake_cv2(webcam))

    source = capture.WebcamSource(stop_wait_s=0.2)
    source.start()
    reader = threading.Thread(target=source.read, daemon=True)
    reader.start()
    assert read_entered.wait(2.0)

    started = time.monotonic()
    source.stop()
    elapsed = time.monotonic() - started
    assert 0.15 <= elapsed < 1.5
    assert events == ["release"]
    never.set()
    reader.join(2.0)


def test_start_applies_io_timeouts(monkeypatch):
    applied = []
    webcam = SimpleNamespace(
        isOpened=lambda: True,
        set=lambda prop, value: applied.append((prop, value)),
        read=lambda: (True, np.zeros((2, 2, 3), dtype=np.uint8)),
        release=lambda: None,
    )
    monkeypatch.setattr(capture, "cv2", _fake_cv2(webcam))

    source = capture.WebcamSource(io_timeout_ms=1234.0)
    source.start()
    source.stop()
    assert (53, 1234.0) in applied
    assert (54, 1234.0) in applied


def test_read_after_stop_raises(monkeypatch):
    webcam = SimpleNamespace(
        isOpened=lambda: True,
        set=lambda *_: None,
        read=lambda: (True, np.zeros((2, 2, 3), dtype=np.uint8)),
        release=lambda: None,
    )
    monkeypatch.setattr(capture, "cv2", _fake_cv2(webcam))
    source = capture.WebcamSource()
    source.start()
    source.stop()
    with pytest.raises(RuntimeError, match="not running"):
        source.read()
