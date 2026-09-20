

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from demo.face_identity import LocalFaceEncoder, SessionFaceTracker, _model_path, _unit
from demo.session_runtime import SessionSelector
from demo.session_state import SessionState
from onevoice.core.models.frame import Frame
from onevoice.core.models.speaker_track import SpeakerTrack

class Scene:
    def __init__(self, **options):
        self.tracks = []
        self.features = {}
        self.inner = SimpleNamespace(process_frame=lambda f: self.tracks, close=Mock())
        self.encoder = SimpleNamespace(encode=lambda f, tracks: self.features)
        self.tracker = SessionFaceTracker(self.inner, self.encoder, **options)
        self.time = 100000

    def frame(self, people, time=None):

        self.time = self.time + 33 if time is None else time
        self.tracks = [
            SpeakerTrack(
                raw, (x, 10, 80, 100), 0.99, {"visible": True, "lip_patch": "original"}
            )
            for raw, x, _ in people
        ]
        self.features = {
            raw: feature for raw, _, feature in people if feature is not None
        }
        return self.tracker.process_frame(Frame(self.time, None, {}))

A, B, C, D = np.eye(4, dtype=np.float32)

def labels(tracks):
    return {t.track_id: t.bounding_box[0] for t in tracks}

def test_three_people_keep_labels_after_track_churn_and_changing_places():
    scene = Scene(confirm_frames=1)
    assert labels(
        scene.frame([("raw0", 0, A), ("raw1", 200, B), ("raw2", 400, C)])
    ) == {
        "person-1": 0,
        "person-2": 200,
        "person-3": 400,
    }

    returned = scene.frame([("raw83", 400, A), ("raw96", 0, C), ("raw101", 200, B)])
    assert labels(returned) == {"person-1": 400, "person-2": 200, "person-3": 0}
    assert all(t.metadata["lip_patch"] == "original" for t in returned)
    assert all("embedding" not in t.metadata for t in returned)

def test_face_returns_after_long_absence_anywhere_with_same_number():
    scene = Scene(confirm_frames=1)
    scene.frame([("old", 0, A)])
    assert scene.frame([], time=110000) == []
    assert labels(scene.frame([("new", 400, A)], time=300000)) == {"person-1": 400}

def test_new_person_at_same_location_does_not_inherit_selected_identity():
    scene = Scene(confirm_frames=1)
    state = SessionState(clock=lambda: 100.0)
    state.begin_start()
    state.mark_running()
    selector = SessionSelector(state)
    frame = Frame(100000, None, {})
    selector.select_target(frame, scene.frame([("same-raw-id", 0, A)]))
    assert state.select("person-1")

    target = selector.select_target(frame, scene.frame([("same-raw-id", 0, B)]))
    assert target.selected_speaker is None
    assert state.selection()[0] is None
    selector.select_target(frame, scene.frame([("other-raw-id", 200, A)]))
    assert state.selection()[0] is None
    assert state.person_name("person-1") == "Person 1"

def test_no_fourth_label_and_no_recycling_absent_peoples_labels():
    scene = Scene(confirm_frames=1)
    scene.frame([("a", 0, A), ("b", 100, B), ("c", 200, C)])
    for i in range(20):
        assert scene.frame([(f"stranger-{i}", 0, D)]) == []
    assert scene.tracker.status()["registered"] == 3
    assert scene.tracker.status()["unmatched"] == 1
    assert labels(scene.frame([("returning", 0, B)])) == {"person-2": 0}

def test_new_profile_requires_consecutive_stable_observations():
    scene = Scene(confirm_frames=3)
    assert scene.frame([("a", 0, A)]) == []
    assert scene.frame([("a", 1, A)]) == []
    assert scene.frame([]) == []
    assert scene.frame([("a", 1, A)]) == []
    assert scene.frame([("a", 1, A)]) == []
    assert labels(scene.frame([("a", 1, A)])) == {"person-1": 1}

def test_reused_video_frame_does_not_confirm_a_new_identity_or_repeat_inference():
    scene = Scene(confirm_frames=3)
    encode = Mock(wraps=scene.encoder.encode)
    scene.encoder.encode = encode
    for _ in range(10):
        assert scene.frame([("a", 0, A)], time=100000) == []
    encode.assert_called_once()
    assert scene.frame([("a", 0, A)], time=100033) == []
    assert labels(scene.frame([("a", 0, A)], time=100066)) == {"person-1": 0}

def test_ambiguous_match_is_withheld_instead_of_switching_or_registering():
    scene = Scene(confirm_frames=1)
    scene.frame([("a", 0, A), ("b", 200, B)])
    assert scene.frame([("uncertain", 100, _unit(A + B))]) == []
    assert scene.tracker.status()["registered"] == 2

    assert scene.frame([("weak", 0, _unit(0.4 * A + 0.9165 * D))]) == []
    assert scene.tracker.status()["registered"] == 2

def test_simultaneous_competing_matches_do_not_duplicate_one_person():
    scene = Scene(confirm_frames=1)
    scene.frame([("a", 0, A)])
    assert scene.frame([("duplicate-a", 0, A), ("duplicate-b", 10, A)]) == []

def test_missing_features_never_fall_back_to_box_position():

    scene = Scene(confirm_frames=1)
    assert scene.frame([("a", 0, None)]) == []
    assert scene.frame([("a", 0, [float("nan")] * 4)]) == []
    assert scene.frame([("a", 0, [0] * 4)]) == []
    scene.encoder.encode = Mock(side_effect=RuntimeError("model failed"))
    assert scene.frame([("a", 0, A)]) == []
    assert scene.tracker.status()["error"]
    assert scene.tracker.status()["registered"] == 0

def test_unreadable_face_coasts_only_on_the_track_that_earned_the_identity():
    scene = Scene(confirm_frames=1, coast_ms=750.0)
    scene.frame([("a", 0, A)])

    held = scene.frame([("a", 0, None)])
    assert labels(held) == {"person-1": 0}
    assert held[0].metadata["identity_coasted"] is True

    assert scene.frame([("b", 0, None)]) == []

def test_a_contradicting_face_breaks_the_coast_immediately():

    scene = Scene(confirm_frames=3, coast_ms=750.0)
    for _ in range(3):
        scene.frame([("a", 0, A)])
    assert labels(scene.frame([("a", 0, A)])) == {"person-1": 0}

    assert scene.frame([("a", 0, D)]) == [], "must not coast onto a different face"

    assert scene.frame([("a", 0, None)]) == []

def test_a_contradicting_face_is_never_labelled_as_the_coasted_person():

    scene = Scene(confirm_frames=1, coast_ms=750.0)
    scene.frame([("a", 0, A)])
    out = scene.frame([("a", 0, D)])
    assert "person-1" not in labels(out)

def test_close_clears_profiles_and_next_session_starts_at_person_one():
    scene = Scene(confirm_frames=1)
    scene.frame([("a", 0, A), ("b", 200, B)])
    scene.tracker.close()
    assert scene.tracker._profiles == []
    assert scene.tracker._pending == {}
    scene.inner.close.assert_called_once()
    assert labels(Scene(confirm_frames=1).frame([("b", 200, B)])) == {"person-1": 200}

def test_person_number_survives_first_observed_order():
    state = SessionState(clock=lambda: 100.0)
    state.begin_start()
    state.observe_tracks(
        [SpeakerTrack("person-3", (0, 0, 1, 1), 1.0, {"person_number": 3})]
    )
    assert state.person_name("person-3") == "Person 3"

def test_encoder_uses_landmark_alignment_and_original_frame(monkeypatch):
    encoder = LocalFaceEncoder.__new__(LocalFaceEncoder)

    face = np.array(
        [[25, 25, 50, 50, 35, 40, 60, 40, 50, 50, 40, 65, 60, 65, 0.99]],
        dtype=np.float32,
    )
    encoder.detector = SimpleNamespace(
        setInputSize=Mock(), detect=Mock(return_value=(1, face))
    )
    encoder.recognizer = SimpleNamespace(
        alignCrop=Mock(return_value="aligned"), feature=Mock(return_value=A)
    )
    frame = Frame(0, np.zeros((720, 1280, 3), dtype=np.uint8), {})
    track = SpeakerTrack("raw", (50, 50, 100, 100), 1.0, {})
    result = encoder.encode(frame, [track, replace(track, track_id="duplicate")])
    assert len(result) == 1
    image_arg, face_arg = encoder.recognizer.alignCrop.call_args.args
    assert image_arg is frame.data
    np.testing.assert_allclose(face_arg[:14], face[0, :14] * 2)
    encoder.recognizer.feature.assert_called_once_with("aligned")

def test_model_cache_requires_valid_checksum_and_rejects_partial_download(
    tmp_path, monkeypatch
):
    import demo.face_identity as module

    model = next(iter(module._MODELS))
    monkeypatch.setattr(
        module.urllib.request, "urlopen", Mock(side_effect=OSError("offline"))
    )
    with pytest.raises(RuntimeError, match="Connect once"):
        _model_path(model, tmp_path)
    assert not list(tmp_path.iterdir())
