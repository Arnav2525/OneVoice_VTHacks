

from __future__ import annotations

import importlib.util

import pytest

from onevoice.streaming.app import build_face_tracker
from onevoice.video.detectors.base import Detection
from onevoice.video.detectors.mediapipe_detector import MediaPipeFaceDetector
from onevoice.video.trackers.stub_tracker import StubFaceTracker

_HAS_MEDIAPIPE = importlib.util.find_spec("mediapipe") is not None

def test_detection_defaults():
    d = Detection(bounding_box=(0, 0, 10, 10), confidence=0.8)
    assert d.landmarks is None
    assert d.lip_bbox is None
    assert d.metadata == {}

@pytest.mark.skipif(_HAS_MEDIAPIPE, reason="mediapipe installed")
def test_mediapipe_detector_raises_without_dependency():
    with pytest.raises(RuntimeError, match="mediapipe"):
        MediaPipeFaceDetector()

def test_build_face_tracker_defaults_to_stub():
    assert isinstance(build_face_tracker({}), StubFaceTracker)
    assert isinstance(
        build_face_tracker({"video": {"tracker": "stub"}}), StubFaceTracker
    )

@pytest.mark.skipif(_HAS_MEDIAPIPE, reason="mediapipe installed")
def test_build_face_tracker_iou_falls_back_loudly_without_mediapipe():
    tracker = build_face_tracker({"video": {"tracker": "iou"}})
    assert isinstance(tracker, StubFaceTracker)

@pytest.mark.skipif(not _HAS_MEDIAPIPE, reason="mediapipe not installed")
def test_build_face_tracker_iou_when_available():
    from onevoice.video.trackers.iou_tracker import IouFaceTracker

    tracker = build_face_tracker({"video": {"tracker": "iou"}})
    assert isinstance(tracker, IouFaceTracker)
