

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from demo.tap_selection import TrackObservingSelector, hit_test, window_click_to_native
from onevoice.core.models.frame import Frame
from onevoice.core.models.speaker_track import SpeakerTrack
from onevoice.core.models.target_selection import TargetSelection

def _track(
    track_id: str,
    box: tuple[float, float, float, float],
    confidence: float = 0.9,
) -> SpeakerTrack:
    return SpeakerTrack(
        track_id=track_id, bounding_box=box, confidence=confidence, metadata={}
    )

class FakeInnerSelector:

    def __init__(self) -> None:
        self.manual_calls: list[str | None] = []
        self.selected_id: str | None = None

    def select_target(
        self, frame: Frame, tracks: list[SpeakerTrack]
    ) -> TargetSelection:
        selected = next((t for t in tracks if t.track_id == self.selected_id), None)
        return TargetSelection(
            timestamp_ms=frame.timestamp_ms, selected_speaker=selected
        )

    def set_manual_target(self, track_id: str | None) -> None:
        self.manual_calls.append(track_id)
        self.selected_id = track_id

    def get_state(self) -> str:
        return "FAKE_STATE"

def test_latest_tracks_starts_empty() -> None:
    selector = TrackObservingSelector(FakeInnerSelector())
    assert selector.latest_tracks() == []

def test_select_target_records_tracks_and_delegates() -> None:
    inner = FakeInnerSelector()
    selector = TrackObservingSelector(inner)
    tracks = [_track("a", (0, 0, 10, 10)), _track("b", (20, 20, 10, 10))]
    frame = Frame(timestamp_ms=1.0, data=None, metadata={})

    result = selector.select_target(frame, tracks)

    assert selector.latest_tracks() == tracks
    assert result.selected_speaker is None

def test_latest_tracks_returns_a_copy_not_a_live_reference() -> None:
    inner = FakeInnerSelector()
    selector = TrackObservingSelector(inner)
    tracks = [_track("a", (0, 0, 10, 10))]
    frame = Frame(timestamp_ms=1.0, data=None, metadata={})
    selector.select_target(frame, tracks)

    got = selector.latest_tracks()
    got.append(_track("mutated-in", (0, 0, 1, 1)))

    assert len(selector.latest_tracks()) == 1

def test_set_manual_target_delegates_to_inner() -> None:
    inner = FakeInnerSelector()
    selector = TrackObservingSelector(inner)

    selector.set_manual_target("a")

    assert inner.manual_calls == ["a"]

def test_set_manual_target_none_delegates_too() -> None:
    inner = FakeInnerSelector()
    selector = TrackObservingSelector(inner)

    selector.set_manual_target("a")
    selector.set_manual_target(None)

    assert inner.manual_calls == ["a", None]

def test_set_manual_target_noop_when_inner_lacks_it() -> None:
    class NoManualSelector:
        def select_target(
            self, frame: Frame, tracks: list[SpeakerTrack]
        ) -> TargetSelection:
            return TargetSelection(
                timestamp_ms=frame.timestamp_ms, selected_speaker=None
            )

    selector = TrackObservingSelector(NoManualSelector())
    selector.set_manual_target("a")

def test_get_state_delegates_when_present() -> None:
    selector = TrackObservingSelector(FakeInnerSelector())
    assert selector.get_state() == "FAKE_STATE"

def test_get_state_none_when_inner_lacks_it() -> None:
    class NoStateSelector:
        def select_target(
            self, frame: Frame, tracks: list[SpeakerTrack]
        ) -> TargetSelection:
            return TargetSelection(
                timestamp_ms=frame.timestamp_ms, selected_speaker=None
            )

    selector = TrackObservingSelector(NoStateSelector())
    assert selector.get_state() is None

def test_end_to_end_manual_selection_reflected_in_next_select_target() -> None:

    inner = FakeInnerSelector()
    selector = TrackObservingSelector(inner)
    tracks = [_track("a", (0, 0, 10, 10)), _track("b", (20, 20, 10, 10))]
    frame = Frame(timestamp_ms=1.0, data=None, metadata={})

    selector.set_manual_target("b")
    result = selector.select_target(frame, tracks)

    assert result.selected_speaker is not None
    assert result.selected_speaker.track_id == "b"

def test_hit_test_empty_tracks_returns_none() -> None:
    assert hit_test([], 5.0, 5.0) is None

def test_hit_test_click_inside_box() -> None:
    tracks = [_track("a", (0, 0, 100, 100))]
    assert hit_test(tracks, 50.0, 50.0) == "a"

def test_hit_test_click_outside_all_boxes() -> None:
    tracks = [_track("a", (0, 0, 100, 100))]
    assert hit_test(tracks, 500.0, 500.0) is None

def test_hit_test_click_on_box_edge_is_inclusive() -> None:
    tracks = [_track("a", (10, 10, 20, 20))]
    assert hit_test(tracks, 10.0, 10.0) == "a"
    assert hit_test(tracks, 30.0, 30.0) == "a"

def test_hit_test_picks_smallest_box_on_overlap() -> None:
    big = _track("big", (0, 0, 200, 200))
    small = _track("small", (50, 50, 20, 20))

    assert hit_test([big, small], 60.0, 60.0) == "small"

    assert hit_test([small, big], 60.0, 60.0) == "small"

def test_hit_test_two_disjoint_faces_picks_correct_one() -> None:
    left = _track("left", (0, 0, 50, 50))
    right = _track("right", (200, 0, 50, 50))

    assert hit_test([left, right], 25.0, 25.0) == "left"
    assert hit_test([left, right], 225.0, 25.0) == "right"
    assert hit_test([left, right], 125.0, 25.0) is None

def test_hit_test_three_disjoint_faces_picks_correct_one() -> None:
    left = _track("left", (0, 0, 50, 50))
    middle = _track("middle", (200, 0, 50, 50))
    right = _track("right", (400, 0, 50, 50))
    faces = [left, middle, right]

    assert hit_test(faces, 25.0, 25.0) == "left"
    assert hit_test(faces, 225.0, 25.0) == "middle"
    assert hit_test(faces, 425.0, 25.0) == "right"
    assert hit_test(faces, 125.0, 25.0) is None
    assert hit_test(faces, 325.0, 25.0) is None

def test_hit_test_three_way_nested_overlap_picks_smallest_regardless_of_order() -> None:

    huge = _track("huge", (0, 0, 300, 300))
    medium = _track("medium", (30, 30, 100, 100))
    tiny = _track("tiny", (50, 50, 20, 20))

    assert hit_test([huge, medium, tiny], 60.0, 60.0) == "tiny"
    assert hit_test([tiny, medium, huge], 60.0, 60.0) == "tiny"
    assert hit_test([medium, huge, tiny], 60.0, 60.0) == "tiny"

def test_hit_test_click_in_gap_among_three_faces_returns_none() -> None:

    a = _track("a", (0, 0, 50, 50))
    b = _track("b", (200, 0, 50, 50))
    c = _track("c", (100, 200, 50, 50))
    assert hit_test([a, b, c], 125.0, 25.0) is None

def test_window_click_to_native_no_scale_is_identity() -> None:
    assert window_click_to_native(100.0, 50.0, 640.0, 480.0, 640.0, 480.0) == (
        100.0,
        50.0,
    )

def test_window_click_to_native_scales_up_proportionally() -> None:

    assert window_click_to_native(200.0, 100.0, 1280.0, 960.0, 640.0, 480.0) == (
        100.0,
        50.0,
    )

def test_window_click_to_native_zero_displayed_size_falls_back_to_raw() -> None:
    assert window_click_to_native(50.0, 60.0, 0.0, 0.0, 640.0, 480.0) == (50.0, 60.0)

def test_window_click_to_native_matches_real_session_clicks() -> None:

    displayed_w, displayed_h = 2048.0, 583.0
    native_w, native_h = 640.0, 480.0
    box_x0, box_y0, box_x1, box_y1 = 240.0, 156.0, 400.0, 324.0

    real_clicks = [(958.0, 334.0), (1007.0, 291.0), (1042.0, 240.0)]
    for raw_x, raw_y in real_clicks:
        nx, ny = window_click_to_native(
            raw_x, raw_y, displayed_w, displayed_h, native_w, native_h
        )
        assert box_x0 <= nx <= box_x1, f"raw=({raw_x},{raw_y}) -> nx={nx} outside box"
        assert box_y0 <= ny <= box_y1, f"raw=({raw_x},{raw_y}) -> ny={ny} outside box"

def test_smoothing_first_observation_is_exact_not_lagged() -> None:

    selector = TrackObservingSelector(FakeInnerSelector(), smoothing_alpha=0.35)
    frame = Frame(timestamp_ms=1.0, data=None, metadata={})

    selector.select_target(frame, [_track("a", (100.0, 100.0, 50.0, 50.0))])

    assert selector.latest_tracks()[0].bounding_box == (100.0, 100.0, 50.0, 50.0)

def test_smoothing_moves_partway_toward_a_jump_not_all_the_way() -> None:
    selector = TrackObservingSelector(FakeInnerSelector(), smoothing_alpha=0.5)
    frame = Frame(timestamp_ms=1.0, data=None, metadata={})

    selector.select_target(frame, [_track("a", (0.0, 0.0, 100.0, 100.0))])

    selector.select_target(frame, [_track("a", (200.0, 0.0, 100.0, 100.0))])

    x = selector.latest_tracks()[0].bounding_box[0]
    assert 0.0 < x < 200.0

def test_smoothing_converges_to_a_steady_position_over_several_frames() -> None:
    selector = TrackObservingSelector(FakeInnerSelector(), smoothing_alpha=0.5)
    frame = Frame(timestamp_ms=1.0, data=None, metadata={})

    selector.select_target(frame, [_track("a", (0.0, 0.0, 100.0, 100.0))])
    for _ in range(30):
        selector.select_target(frame, [_track("a", (200.0, 0.0, 100.0, 100.0))])

    x = selector.latest_tracks()[0].bounding_box[0]
    assert x == pytest.approx(200.0, abs=0.5)

def test_smoothing_does_not_affect_what_the_inner_selector_receives() -> None:

    inner = FakeInnerSelector()
    selector = TrackObservingSelector(inner, smoothing_alpha=0.5)
    frame = Frame(timestamp_ms=1.0, data=None, metadata={})
    seen_by_inner: list[list[SpeakerTrack]] = []

    class RecordingInner(FakeInnerSelector):
        def select_target(
            self, frame: Frame, tracks: list[SpeakerTrack]
        ) -> TargetSelection:
            seen_by_inner.append(list(tracks))
            return super().select_target(frame, tracks)

    selector = TrackObservingSelector(RecordingInner(), smoothing_alpha=0.5)
    selector.select_target(frame, [_track("a", (0.0, 0.0, 100.0, 100.0))])
    selector.select_target(frame, [_track("a", (200.0, 0.0, 100.0, 100.0))])

    assert seen_by_inner[1][0].bounding_box == (200.0, 0.0, 100.0, 100.0)

def test_smoothing_forgets_a_track_that_disappears_then_reappears() -> None:

    selector = TrackObservingSelector(FakeInnerSelector(), smoothing_alpha=0.5)
    frame = Frame(timestamp_ms=1.0, data=None, metadata={})

    selector.select_target(frame, [_track("a", (0.0, 0.0, 50.0, 50.0))])
    selector.select_target(frame, [])
    selector.select_target(frame, [_track("a", (900.0, 900.0, 50.0, 50.0))])

    assert selector.latest_tracks()[0].bounding_box == (900.0, 900.0, 50.0, 50.0)

def test_smoothing_handles_multiple_independent_tracks() -> None:
    selector = TrackObservingSelector(FakeInnerSelector(), smoothing_alpha=0.5)
    frame = Frame(timestamp_ms=1.0, data=None, metadata={})

    selector.select_target(
        frame,
        [_track("a", (0.0, 0.0, 50.0, 50.0)), _track("b", (500.0, 0.0, 50.0, 50.0))],
    )
    selector.select_target(
        frame,
        [
            _track("a", (100.0, 0.0, 50.0, 50.0)),
            _track("b", (500.0, 0.0, 50.0, 50.0)),
        ],
    )

    by_id = {t.track_id: t for t in selector.latest_tracks()}
    assert by_id["a"].bounding_box[0] == pytest.approx(50.0)
    assert by_id["b"].bounding_box[0] == pytest.approx(500.0)

def test_smoothing_handles_three_simultaneous_tracks_independently() -> None:

    selector = TrackObservingSelector(FakeInnerSelector(), smoothing_alpha=0.5)
    frame = Frame(timestamp_ms=1.0, data=None, metadata={})

    selector.select_target(
        frame,
        [
            _track("a", (0.0, 0.0, 50.0, 50.0)),
            _track("b", (500.0, 0.0, 50.0, 50.0)),
            _track("c", (0.0, 500.0, 50.0, 50.0)),
        ],
    )

    selector.select_target(
        frame,
        [
            _track("a", (100.0, 0.0, 50.0, 50.0)),
            _track("b", (500.0, 0.0, 50.0, 50.0)),
            _track("c", (0.0, 500.0, 50.0, 50.0)),
        ],
    )
    selector.select_target(
        frame,
        [
            _track("a", (100.0, 0.0, 50.0, 50.0)),
            _track("b", (500.0, 0.0, 50.0, 50.0)),
        ],
    )

    tracks = selector.latest_tracks()
    by_id = {t.track_id: t for t in tracks}
    assert set(by_id) == {"a", "b"}

    assert by_id["a"].bounding_box[0] == pytest.approx(75.0)
    assert by_id["b"].bounding_box[0] == pytest.approx(
        500.0
    )
