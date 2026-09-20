

from __future__ import annotations

import numpy as np

from onevoice.core.models.frame import Frame
from onevoice.video.detectors.base import Detection
from onevoice.video.trackers.iou_tracker import (
    LIP_PATCH_SIZE,
    IouFaceTracker,
    TrackerConfig,
    _crop_lip_patch,
    iou,
)

class ScriptedDetector:

    def __init__(self, script: list[list[Detection] | str]) -> None:
        self._script = script
        self._i = 0

    def detect(self, frame: Frame) -> list[Detection]:
        item = self._script[min(self._i, len(self._script) - 1)]
        self._i += 1
        if item == "raise":
            raise RuntimeError("boom")
        return list(item)  # type: ignore[arg-type]

def _frame(ts: float) -> Frame:
    return Frame(timestamp_ms=ts, data=None, metadata={"width": 640, "height": 480})

def _det(x, y, w=100.0, h=100.0, conf=0.9, mouth=None):
    landmarks = {"mouth": mouth} if mouth else None
    lip = (mouth[0] - 10, mouth[1] - 5, 20.0, 10.0) if mouth else None
    return Detection((x, y, w, h), conf, landmarks=landmarks, lip_bbox=lip)

def test_iou_basic():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (100, 100, 10, 10)) == 0.0
    assert 0.0 < iou((0, 0, 10, 10), (5, 0, 10, 10)) < 1.0

def test_confirms_after_min_hits_and_stable_id():
    det = _det(100, 100)
    tracker = IouFaceTracker(
        ScriptedDetector([[det]]), TrackerConfig(min_hits=3, max_age=15)
    )

    assert tracker.process_frame(_frame(0.0)) == []
    assert tracker.process_frame(_frame(33.0)) == []
    out = tracker.process_frame(_frame(66.0))
    assert len(out) == 1
    tid = out[0].track_id

    for i in range(3, 8):
        out = tracker.process_frame(_frame(33.0 * i))
        assert len(out) == 1
        assert out[0].track_id == tid

def test_two_faces_keep_distinct_ids():
    a, b = _det(50, 50), _det(400, 300)
    tracker = IouFaceTracker(
        ScriptedDetector([[a, b]]), TrackerConfig(min_hits=1)
    )
    out = tracker.process_frame(_frame(0.0))
    ids = {t.track_id for t in out}
    assert len(ids) == 2
    out2 = tracker.process_frame(_frame(33.0))
    assert {t.track_id for t in out2} == ids

def test_id_not_reused_after_permanent_loss():
    present = _det(100, 100)

    script: list = [[present]]
    tracker = IouFaceTracker(
        ScriptedDetector(script),
        TrackerConfig(min_hits=1, max_age=2, reid_window_ms=50.0),
    )
    first = tracker.process_frame(_frame(0.0))[0].track_id

    tracker._detector = ScriptedDetector([[]])  # type: ignore[attr-defined]
    for i in range(1, 8):
        tracker.process_frame(_frame(100.0 * i))

    tracker._detector = ScriptedDetector([[present]])  # type: ignore[attr-defined]
    out = tracker.process_frame(_frame(2000.0))
    assert len(out) == 1
    assert out[0].track_id != first

def test_reid_recovers_identity_within_window():
    present = _det(100, 100)
    tracker = IouFaceTracker(
        ScriptedDetector([[present]]),
        TrackerConfig(min_hits=1, max_age=1, reid_window_ms=1000.0),
    )
    tid = tracker.process_frame(_frame(0.0))[0].track_id

    tracker._detector = ScriptedDetector([[]])  # type: ignore[attr-defined]
    tracker.process_frame(_frame(33.0))
    tracker.process_frame(_frame(66.0))

    tracker._detector = ScriptedDetector([[present]])  # type: ignore[attr-defined]
    out = tracker.process_frame(_frame(120.0))
    assert len(out) == 1
    assert out[0].track_id == tid
    assert tracker.get_stats()["reid_recoveries"] == 1

def test_detector_failure_does_not_crash_and_coasts():
    det = _det(100, 100)
    tracker = IouFaceTracker(
        ScriptedDetector([[det], [det], [det], "raise"]),
        TrackerConfig(min_hits=1),
    )
    for i in range(3):
        tracker.process_frame(_frame(33.0 * i))

    out = tracker.process_frame(_frame(200.0))
    assert isinstance(out, list)

def test_metadata_carries_lip_and_lifecycle_fields():
    det = _det(100, 100, mouth=(150.0, 170.0))
    tracker = IouFaceTracker(ScriptedDetector([[det]]), TrackerConfig(min_hits=1))
    track = tracker.process_frame(_frame(42.0))[0]
    md = track.metadata
    assert md["tracker"] == "iou"
    assert md["visible"] is True
    assert md["last_seen_ms"] == 42.0
    assert md["lip_roi_ts_ms"] == 42.0
    assert "lip_roi" in md and "landmarks" in md
    assert 0.0 <= track.confidence <= 1.0

def _frame_with_image(ts: float, image: np.ndarray) -> Frame:
    h, w = image.shape[:2]
    return Frame(timestamp_ms=ts, data=image, metadata={"width": w, "height": h})

def test_metadata_lip_patch_is_real_resized_2d_crop_when_frame_has_data():
    det = _det(100, 100, mouth=(150.0, 170.0))
    tracker = IouFaceTracker(ScriptedDetector([[det]]), TrackerConfig(min_hits=1))
    image = np.tile(np.arange(480, dtype=np.uint8).reshape(-1, 1), (1, 640))
    track = tracker.process_frame(_frame_with_image(42.0, image))[0]
    patch = track.metadata.get("lip_patch")
    assert patch is not None
    assert isinstance(patch, np.ndarray)
    assert patch.shape == (LIP_PATCH_SIZE, LIP_PATCH_SIZE)

    assert -3.0 < float(patch.min()) < 4.0
    assert -3.0 < float(patch.max()) < 4.0
    assert float(patch.max()) > float(patch.min())

def test_attach_frame_bgr_copies_shared_frame_into_metadata():
    det = _det(100, 100, mouth=(150.0, 170.0))
    tracker = IouFaceTracker(
        ScriptedDetector([[det]]),
        TrackerConfig(min_hits=1, attach_frame_bgr=True),
    )
    image = np.zeros((40, 50, 3), dtype=np.uint8)
    image[10, 20] = (1, 2, 3)
    track = tracker.process_frame(_frame_with_image(1.0, image))[0]
    frame_bgr = track.metadata.get("frame_bgr")
    assert frame_bgr is not None
    assert frame_bgr.shape == (40, 50, 3)
    assert tuple(frame_bgr[10, 20]) == (1, 2, 3)

    image[10, 20] = (9, 9, 9)
    assert tuple(frame_bgr[10, 20]) == (1, 2, 3)

def test_crop_lip_patch_resizes_arbitrary_crop_to_lip_patch_size():
    image = (np.arange(40 * 20, dtype=np.float32).reshape(40, 20) % 255).astype(
        np.uint8
    )
    frame = _frame_with_image(0.0, image)
    patch = _crop_lip_patch(frame, (0.0, 0.0, 20.0, 40.0))
    assert patch is not None
    assert patch.shape == (LIP_PATCH_SIZE, LIP_PATCH_SIZE)

def test_crop_lip_patch_returns_none_for_degenerate_bbox():
    image = np.zeros((100, 100), dtype=np.uint8)
    frame = _frame_with_image(0.0, image)
    assert _crop_lip_patch(frame, (10.0, 10.0, 0.0, 0.0)) is None
    assert _crop_lip_patch(None, (0.0, 0.0, 10.0, 10.0)) is None

    assert _crop_lip_patch(frame, (200.0, 200.0, 50.0, 50.0)) is None

def test_crop_lip_patch_pads_out_of_bounds_box_instead_of_clipping():

    image = np.full((100, 100), 200, dtype=np.uint8)
    frame = _frame_with_image(0.0, image)

    patch = _crop_lip_patch(frame, (80.0, 80.0, 50.0, 50.0))
    assert patch is not None
    assert patch.shape == (LIP_PATCH_SIZE, LIP_PATCH_SIZE)

    black = (0.0 / 255.0 - 0.421) / 0.165
    bright = (200.0 / 255.0 - 0.421) / 0.165

    assert abs(float(patch[0, 0]) - bright) < 0.2
    assert abs(float(patch[-1, -1]) - black) < 0.2

def test_get_stats_is_snapshot():
    det = _det(100, 100)
    tracker = IouFaceTracker(ScriptedDetector([[det]]), TrackerConfig(min_hits=1))
    tracker.process_frame(_frame(0.0))
    s = tracker.get_stats()
    assert s["frames_processed"] == 1
    assert s["tracks_created"] == 1

    tracker.process_frame(_frame(33.0))
    assert s["frames_processed"] == 1
