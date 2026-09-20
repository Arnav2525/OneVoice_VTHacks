

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from demo.session_recording import RecordingSnapshot
from demo.session_runtime import PreviewTracker
from demo.session_state import SessionView
from demo.session_view import face_at, inside, render_session
from demo.tap_to_select import SessionWindow

def view(phase="listening", *, title="Select someone to hear", detail="Preview only"):
    return SessionView(
        phase=phase,
        title=title,
        detail=detail,
        mode="preview",
        active=True,
        selected_id=None,
        person=None,
        input_level=0.25,
        output_level=0.0,
        elapsed_s=5.0,
        face_count=3,
    )

@pytest.mark.parametrize("size", [(960, 640), (1280, 800), (1920, 1080)])
def test_letterboxed_camera_maps_all_three_faces_and_excludes_sidebar(size):
    tracks = PreviewTracker().process_frame(None)
    frame = SimpleNamespace(data=np.zeros((480, 640, 3), dtype=np.uint8))
    canvas, controls, rect, native = render_session(
        view(),
        frame,
        tracks,
        lambda name: name,
        RecordingSnapshot(False, 0, None, None),
        size=size,
    )
    assert canvas.shape == (size[1], size[0], 3)
    rx, ry, rw, rh = rect
    for track in tracks:
        x, y, w, h = track.bounding_box
        click = (rx + (x + w / 2) * rw / 640, ry + (y + h / 2) * rh / 480)
        assert face_at(*click, rect, native, tracks) == track.track_id
    assert face_at(size[0] - 100, 200, rect, native, tracks) is None
    for button in controls.values():
        x, y, w, h = button
        assert face_at(x + w / 2, y + h / 2, rect, native, tracks) is None
        assert 0 <= x < x + w <= size[0]
        assert 0 <= y < y + h <= size[1]
    if ry > 106:
        assert face_at(rx + 50, 106, rect, native, tracks) is None

def test_coasting_tracks_cannot_be_clicked():
    track = PreviewTracker().process_frame(None)[0]
    track = replace(track, metadata={"visible": False})
    assert face_at(100, 200, (0, 0, 640, 480), (640, 480), [track]) is None

def test_long_error_stays_inside_sidebar_text_area():
    empty = RecordingSnapshot(False, 0, None, None)
    baseline, _, _, _ = render_session(
        view("error", detail=""),
        None,
        [],
        str,
        empty,
        size=(960, 640),
    )
    long_error, _, _, _ = render_session(
        view("error", detail=("Device unavailable: " + "x" * 200 + " ") * 20),
        None,
        [],
        str,
        empty,
        size=(960, 640),
    )
    changed_y, changed_x = np.nonzero(np.any(baseline != long_error, axis=2))
    assert changed_y.size > 0
    assert changed_y.max() < 640 - 280 - 20
    assert changed_x.min() >= 960 - 285
    assert changed_x.max() <= 960 - 40

def test_actual_window_buttons_route_actions_and_background_clicks_do_nothing():
    runner = Mock(busy=False)
    app = SessionWindow(runner, windowed=True)
    tracks = PreviewTracker().process_frame(None)
    _, app.controls, app.video_rect, app.native_size = render_session(
        view(),
        None,
        tracks,
        str,
        RecordingSnapshot(False, 0, None, None),
    )
    app.drawn_tracks = tracks
    for name, method in (
        ("session", runner.start),
        ("record", runner.toggle_recording),
    ):
        x, y, w, h = app.controls[name]
        app.click(x + w / 2, y + h / 2)
        method.assert_called_once()
    runner.busy = True
    app.action("session")
    runner.stop.assert_called_once()
    app.action("clear")
    runner.state.select.assert_called_once_with(None)
    runner.state.select.reset_mock()
    app.click(5, 5)
    runner.state.select.assert_not_called()
    assert not inside(-1, -1, app.controls["session"])

def test_quitting_prevents_new_actions():
    runner = Mock(busy=False)
    app = SessionWindow(runner)
    app.quitting = True
    app.action("session")
    app.action("record")
    runner.start.assert_not_called()
    runner.toggle_recording.assert_not_called()
