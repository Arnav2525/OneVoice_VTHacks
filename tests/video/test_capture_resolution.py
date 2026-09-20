"""WebcamSource resolution negotiation -- no camera needed.

cv2.VideoCapture.set() only *requests* a mode; a driver that can't honour it
silently substitutes another. These tests use a fake cv2 to pin down that
(a) the size the driver actually agreed to is what frames are stamped with,
and (b) a refusal is logged rather than passing unnoticed.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from onevoice.video import capture  # noqa: E402

CAP_W, CAP_H, CAP_FPS = 3, 4, 5  # arbitrary stand-ins for cv2.CAP_PROP_*


class _FakeCapture:
    """A camera whose driver grants ``supported`` no matter what is asked."""

    def __init__(self, supported: tuple[int, int]) -> None:
        self._size = supported
        self.requested: dict[int, float] = {}

    def isOpened(self) -> bool:  # noqa: N802 - cv2 API
        return True

    def set(self, prop: int, value: float) -> bool:
        self.requested[prop] = value
        return True  # cv2 returns True even when the driver ignores the value

    def get(self, prop: int) -> float:
        return float(self._size[0] if prop == CAP_W else self._size[1])

    def read(self):
        import numpy as np

        return True, np.zeros((self._size[1], self._size[0], 3), dtype=np.uint8)

    def release(self) -> None:
        pass


def _install_fake_cv2(monkeypatch: pytest.MonkeyPatch, supported: tuple[int, int]):
    fake = _FakeCapture(supported)

    class _Cv2:
        CAP_PROP_FRAME_WIDTH = CAP_W
        CAP_PROP_FRAME_HEIGHT = CAP_H
        CAP_PROP_FPS = CAP_FPS

        @staticmethod
        def VideoCapture(_index: int) -> _FakeCapture:  # noqa: N802
            return fake

    monkeypatch.setattr(capture, "cv2", _Cv2)
    return fake


def test_requested_size_is_sent_to_the_driver(monkeypatch) -> None:
    fake = _install_fake_cv2(monkeypatch, supported=(1280, 720))
    source = capture.WebcamSource(width=1280, height=720)
    source.start()
    assert fake.requested[CAP_W] == 1280
    assert fake.requested[CAP_H] == 720


def test_honoured_request_stamps_frames_with_that_size(monkeypatch, caplog) -> None:
    _install_fake_cv2(monkeypatch, supported=(1280, 720))
    source = capture.WebcamSource(width=1280, height=720)
    with caplog.at_level(logging.INFO, logger="onevoice.video.capture"):
        source.start()
    frame = source.read()
    assert (frame.metadata["width"], frame.metadata["height"]) == (1280, 720)
    assert frame.data.shape[:2] == (720, 1280)
    assert not any(r.levelno >= logging.WARNING for r in caplog.records)


def test_refused_request_reports_the_real_size_not_the_requested_one(
    monkeypatch, caplog
) -> None:
    """The regression this exists for: ask for 720p, driver gives 480p. The
    metadata must describe the frame it is attached to, and the mismatch
    must be logged -- otherwise it just looks like a wrong-aspect UI bug."""
    _install_fake_cv2(monkeypatch, supported=(640, 480))
    source = capture.WebcamSource(width=1280, height=720)
    with caplog.at_level(logging.WARNING, logger="onevoice.video.capture"):
        source.start()
    frame = source.read()
    assert (frame.metadata["width"], frame.metadata["height"]) == (640, 480)
    assert frame.data.shape[:2] == (480, 640)
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("asked for 1280x720" in m and "gave 640x480" in m for m in warnings)


def test_default_size_is_unchanged_640x480(monkeypatch) -> None:
    # Unconfigured runs must behave exactly as they always did.
    fake = _install_fake_cv2(monkeypatch, supported=(640, 480))
    capture.WebcamSource().start()
    assert (fake.requested[CAP_W], fake.requested[CAP_H]) == (640, 480)
