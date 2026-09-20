

from __future__ import annotations

import array
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from demo.preflight import (  # noqa: E402
    _FAIL,
    _PASS,
    _WARN,
    Report,
    _check_config,
    _check_face_tracker,
    _check_microphone,
    _check_pipeline_dry_run,
    _check_separator,
    _check_target_selector,
    _check_webcam,
    _rms,
    _run,
)

def test_rms_of_silence_is_zero() -> None:
    assert _rms(array.array("f", [0.0] * 100)) == 0.0

def test_rms_of_empty_array_is_zero_not_a_crash() -> None:

    assert _rms(array.array("f", [])) == 0.0

def test_rms_of_constant_tone_matches_amplitude() -> None:
    assert _rms(array.array("f", [0.5] * 1000)) == pytest.approx(0.5, abs=1e-6)

def test_report_ok_is_true_with_only_pass_and_warn() -> None:
    report = Report()
    report.add("a", _PASS, "fine")
    report.add("b", _WARN, "hmm")
    assert report.ok is True

def test_report_ok_is_false_after_any_fail() -> None:
    report = Report()
    report.add("a", _PASS, "fine")
    report.add("b", _FAIL, "broken")
    assert report.ok is False

def test_run_catches_exceptions_as_fail_instead_of_crashing() -> None:
    report = Report()

    def _boom() -> tuple[str, str, None]:
        raise RuntimeError("camera on fire")

    result = _run(report, "webcam", _boom)
    assert result is None
    assert report.ok is False

def test_check_config_defaults_report_passthrough_stub_first_track() -> None:
    status, detail, config = _check_config(None)
    assert status == _PASS
    assert config == {}
    assert "backend=passthrough" in detail
    assert "tracker=stub" in detail
    assert "selection=first_track" in detail

def test_check_separator_default_config_is_warn_not_fail() -> None:

    status, detail, separator = _check_separator({})
    assert status == _WARN
    assert "passthrough" in detail
    assert separator is not None

def test_check_separator_fails_when_real_backend_requested_but_degrades() -> None:
    config = {"backend": {"name": "not_a_real_backend_xyz"}}
    status, detail, separator = _check_separator(config)
    assert status == _FAIL
    assert "NO REAL SEPARATION" in detail
    assert separator is not None

def test_check_face_tracker_stub_config_passes() -> None:
    status, _detail, tracker = _check_face_tracker({})
    assert status == _PASS
    assert type(tracker).__name__ == "StubFaceTracker"

def test_check_face_tracker_fails_when_iou_requested_but_mediapipe_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import demo.preflight as preflight_mod

    def _fake_build_face_tracker(_config: dict) -> object:
        from onevoice.video.trackers.stub_tracker import StubFaceTracker

        return StubFaceTracker()

    monkeypatch.setattr(preflight_mod, "build_face_tracker", _fake_build_face_tracker)
    status, detail, _tracker = _check_face_tracker({"video": {"tracker": "iou"}})
    assert status == _FAIL
    assert "DISABLED" in detail

def test_check_target_selector_default_is_first_track() -> None:
    status, detail, selector = _check_target_selector({})
    assert status == _PASS
    assert detail == "FirstTrackSelector"
    assert selector is not None

def test_check_microphone_mock_mode_passes_without_hardware() -> None:
    status, detail, _payload = _check_microphone({}, live=False)
    assert status == _PASS
    assert "mock source OK" in detail

def test_check_webcam_mock_mode_passes_without_hardware() -> None:
    status, detail, _payload = _check_webcam(live=False)
    assert status == _PASS
    assert "mock source OK" in detail

def test_check_pipeline_dry_run_mock_mode_produces_results() -> None:
    config: dict = {}
    _status, _detail, separator = _check_separator(config)
    _status2, _detail2, selector = _check_target_selector(config)
    status, detail, _payload = _check_pipeline_dry_run(
        config, live=False, separator=separator, selector=selector, run_seconds=0.5
    )
    assert status == _PASS
    assert "results_processed=" in detail
    assert "results_processed=0 " not in detail

class _LazyRealSeparator:

    def __init__(self, load_error: Exception | None = None) -> None:
        self._loaded = False
        self._load_error = load_error

    def load(self) -> None:
        if self._load_error is not None:
            raise self._load_error
        self._loaded = True

    def get_status(self) -> dict:
        return {"backend": "dolphin", "is_real_separation": self._loaded}

def test_check_separator_loads_lazy_backend_before_judging_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    import demo.preflight as preflight_mod

    monkeypatch.setattr(
        preflight_mod, "build_separator", lambda _config: _LazyRealSeparator()
    )
    status, detail, separator = _check_separator({"backend": {"name": "dolphin"}})
    assert status == _PASS
    assert "real_separation=True" in detail
    assert separator is not None

def test_check_separator_fails_with_the_load_error_when_loading_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    import demo.preflight as preflight_mod

    monkeypatch.setattr(
        preflight_mod,
        "build_separator",
        lambda _config: _LazyRealSeparator(RuntimeError("CUDA out of memory")),
    )
    status, detail, separator = _check_separator({"backend": {"name": "dolphin"}})
    assert status == _FAIL
    assert "NO REAL SEPARATION" in detail
    assert "CUDA out of memory" in detail
    assert separator is not None

def test_check_separator_handles_backend_without_a_load_method() -> None:

    status, detail, separator = _check_separator({"backend": {"name": "passthrough"}})
    assert status == _WARN
    assert "passthrough" in detail
    assert separator is not None
