

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("cv2")

from demo.face_identity import SessionFaceTracker
from onevoice.core.models.frame import Frame
from onevoice.core.models.speaker_track import SpeakerTrack

class _Tracker:
    def __init__(self, tracks):
        self.tracks = tracks

    def process_frame(self, frame):
        return list(self.tracks)

class _Encoder:

    def __init__(self):
        self.features = {}

    def encode(self, frame, tracks):
        return dict(self.features)

def _track(track_id="t1", x=0):
    return SpeakerTrack(track_id, (x, 0, 100, 100), 1.0, {"visible": True})

def _vec(*values):
    v = np.array(values, dtype=np.float32)
    return v / np.linalg.norm(v)

def _register(tracker, encoder, feature, *, track_id="t1", start_ms=0.0):

    ts = start_ms
    for _ in range(tracker.confirm_frames):
        encoder.features = {track_id: feature}
        out = tracker.process_frame(Frame(ts, np.zeros((4, 4, 3)), {}))
        ts += 33.0
    return ts, out

def _setup():
    encoder = _Encoder()
    tracker = SessionFaceTracker(_Tracker([_track()]), encoder, confirm_frames=3)
    ts, out = _register(tracker, encoder, _vec(1, 0, 0))
    assert [t.track_id for t in out] == ["person-1"]
    return tracker, encoder, ts

def test_person_survives_a_frame_the_encoder_could_not_read():

    tracker, encoder, ts = _setup()
    encoder.features = {}
    out = tracker.process_frame(Frame(ts, np.zeros((4, 4, 3)), {}))
    assert [t.track_id for t in out] == ["person-1"]
    assert out[0].metadata["identity_coasted"] is True
    assert out[0].metadata["person_number"] == 1

def test_person_survives_a_similarity_dip_from_a_head_turn():
    tracker, encoder, ts = _setup()

    encoder.features = {"t1": _vec(1.0, 2.3, 0.0)}
    out = tracker.process_frame(Frame(ts, np.zeros((4, 4, 3)), {}))
    assert [t.track_id for t in out] == ["person-1"]
    assert out[0].metadata["identity_coasted"] is True

def test_coast_expires_so_a_stale_identity_cannot_persist():
    tracker, encoder, ts = _setup()
    encoder.features = {}
    late = Frame(ts + tracker.coast_ms + 1.0, np.zeros((4, 4, 3)), {})
    out = tracker.process_frame(late)
    assert out == []

def test_a_confident_match_outranks_a_coast():
    tracker, encoder, ts = _setup()

    tracker.tracker.tracks = [_track("t1"), _track("t2", x=300)]
    ts, _ = _register(tracker, encoder, _vec(0, 1, 0), track_id="t2", start_ms=ts)
    assert len(tracker._profiles) == 2

    encoder.features = {"t2": _vec(0, 1, 0)}
    out = tracker.process_frame(Frame(ts, np.zeros((4, 4, 3)), {}))
    labels = sorted(t.track_id for t in out)
    assert labels == ["person-1", "person-2"]
    by_label = {t.track_id: t for t in out}
    assert by_label["person-2"].metadata["raw_track_id"] == "t2"
    assert by_label["person-2"].metadata["identity_coasted"] is False
    assert by_label["person-1"].metadata["raw_track_id"] == "t1"
    assert by_label["person-1"].metadata["identity_coasted"] is True

def test_coasting_never_transfers_an_identity_to_a_new_track():

    tracker, encoder, ts = _setup()
    tracker.tracker.tracks = [_track("stranger", x=400)]
    encoder.features = {}
    out = tracker.process_frame(Frame(ts, np.zeros((4, 4, 3)), {}))
    assert out == []

def test_a_confident_frame_refreshes_the_coast_window():
    tracker, encoder, ts = _setup()
    for _ in range(40):
        encoder.features = {"t1": _vec(1, 0, 0)}
        out = tracker.process_frame(Frame(ts, np.zeros((4, 4, 3)), {}))
        ts += 100.0
        assert [t.track_id for t in out] == ["person-1"]
        assert out[0].metadata["identity_coasted"] is False
    encoder.features = {}
    out = tracker.process_frame(Frame(ts, np.zeros((4, 4, 3)), {}))
    assert [t.track_id for t in out] == ["person-1"]

def test_status_reports_coasted_count():
    tracker, encoder, ts = _setup()
    encoder.features = {}
    tracker.process_frame(Frame(ts, np.zeros((4, 4, 3)), {}))
    assert tracker.status()["coasted"] == 1
