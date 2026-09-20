

from __future__ import annotations

import copy
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from demo import devices  # noqa: E402

def row(name, host=0, inputs=1, outputs=0):
    return dict(
        name=name,
        hostapi=host,
        max_input_channels=inputs,
        max_output_channels=outputs,
        default_samplerate=48000.0,
    )

@pytest.fixture
def backend(monkeypatch):
    backend = SimpleNamespace(
        query_hostapis=Mock(
            return_value=[
                {"name": "MME"},
                {"name": "Windows WASAPI"},
                {"name": "Windows DirectSound"},
            ]
        ),
        query_devices=Mock(
            return_value=[
                row("Laptop microphone"),
                row("Microphone (HD Webcam C270)"),
                row("Microphone (HD Webcam C270)", host=1),
                row("Microphone (HD Webcam C270)", host=2),
                row("Wired headphones", inputs=0, outputs=2),
            ]
        ),
        check_input_settings=Mock(),
        check_output_settings=Mock(),
        InputStream=Mock(side_effect=AssertionError("capture must not open")),
        OutputStream=Mock(side_effect=AssertionError("playback must not open")),
    )
    monkeypatch.setattr(devices, "_audio_backend", lambda: backend)
    return backend

def test_list_reports_directions_and_never_opens_streams(backend):
    inventory = devices.list_audio_devices()
    assert len(inventory) == 5
    text = devices.format_device_list(inventory)
    assert "2: Microphone (HD Webcam C270) [Windows WASAPI]" in text
    assert "4: Wired headphones [MME]" in text
    assert "--input-device C270" in text
    backend.check_input_settings.assert_not_called()
    backend.InputStream.assert_not_called()
    backend.OutputStream.assert_not_called()

@pytest.mark.parametrize("value", ["C270", "c270", "LOGITECH", "HD WEBCAM C270"])
def test_windows_aliases_resolve_same_microphone(backend, value):
    assert devices.resolve_audio_device(value, kind="input") == 2
    backend.check_input_settings.assert_called_once_with(
        device=2,
        samplerate=16000,
        channels=1,
        dtype="float32",
    )
    backend.InputStream.assert_not_called()

def test_unsupported_format_tries_other_alias_only(backend):
    def support(*, device, **kwargs):
        if device == 2:
            raise RuntimeError("invalid sample rate")

    backend.check_input_settings.side_effect = support
    assert devices.resolve_audio_device("C270", kind="input") == 3
    checked = [
        call.kwargs["device"] for call in backend.check_input_settings.call_args_list
    ]
    assert checked == [2, 3]

def test_numeric_id_is_exact_and_output_direction_is_validated(backend):
    assert devices.resolve_audio_device("4", kind="output") == 4
    with pytest.raises(ValueError, match="was not found"):
        devices.resolve_audio_device(4, kind="input")
    backend.check_input_settings.side_effect = RuntimeError("unsupported")
    with pytest.raises(ValueError, match="does not support"):
        devices.resolve_audio_device("2", kind="input")
    assert backend.check_input_settings.call_count == 1

def test_missing_microphone_never_substitutes_laptop(backend):
    backend.query_devices.return_value = [row("Laptop microphone")]
    with pytest.raises(ValueError, match="system default will not be substituted"):
        devices.resolve_audio_device("C270", kind="input")
    backend.check_input_settings.assert_not_called()

def test_multiple_physical_devices_require_explicit_id(backend):
    backend.query_devices.return_value.append(row("Microphone (HD Webcam C270 #2)"))
    with pytest.raises(ValueError, match="More than one"):
        devices.resolve_audio_device("C270", kind="input")
    assert devices.resolve_audio_device(5, kind="input") == 5

def test_same_name_twice_within_hostapi_is_ambiguous(backend):
    backend.query_devices.return_value.append(row("Microphone (HD Webcam C270)"))
    with pytest.raises(ValueError, match="exact numeric ID"):
        devices.resolve_audio_device("C270", kind="input")

def test_explicit_hostapi_disambiguates(backend):
    assert devices.resolve_audio_device("C270 MME", kind="input") == 1

def test_windows_truncated_name_is_alias(backend):
    full = "Microphone (Logitech HD Webcam C270)"
    backend.query_devices.return_value = [row(full[:31]), row(full, host=1)]
    assert devices.resolve_audio_device("Logitech", kind="input") == 1

@pytest.mark.parametrize("value", [None, "default", " DEFAULT "])
def test_intentional_default_does_not_enumerate(backend, value):
    assert devices.resolve_audio_device(value, kind="input") is None
    backend.query_devices.assert_not_called()

@pytest.mark.parametrize("value", [-1, "-1", "", " ", True, 1.5])
def test_invalid_selection_is_rejected_before_enumeration(backend, value):
    with pytest.raises(ValueError):
        devices.resolve_audio_device(value, kind="input")
    backend.query_devices.assert_not_called()

def test_cli_list_does_not_create_session(backend, monkeypatch, capsys):
    from demo import tap_to_select

    session = Mock(side_effect=AssertionError("no session during inventory"))
    monkeypatch.setattr(tap_to_select, "SessionRunner", session)
    assert tap_to_select.main(["--list-devices"]) == 0
    assert "HD Webcam C270" in capsys.readouterr().out
    session.assert_not_called()

def test_cli_overrides_are_passed_to_session(backend, monkeypatch):
    from demo import tap_to_select, web_session

    source = {
        "audio": {"sample_rate": 16000, "input_device": "C270"},
        "video": {"device_index": 1},
    }
    original = copy.deepcopy(source)
    monkeypatch.setattr(
        tap_to_select,
        "load_experiment_config",
        lambda _: copy.deepcopy(source),
    )
    session = Mock()
    monkeypatch.setattr(tap_to_select, "SessionRunner", session)
    web = Mock(return_value=0)
    monkeypatch.setattr(web_session, "run_web", web)
    assert (
        tap_to_select.main(
            [
                "--live",
                "--camera-index",
                "2",
                "--input-device",
                "5",
                "--output-device",
                "default",
            ]
        )
        == 0
    )
    config = session.call_args.args[0]
    assert config["video"]["device_index"] == 2
    assert config["audio"]["input_device"] == "5"
    assert config["audio"]["output_device"] == "default"
    assert source == original
    backend.query_devices.assert_not_called()

def test_cli_rejects_negative_camera_index():
    from demo import tap_to_select

    with pytest.raises(SystemExit) as exc:
        tap_to_select.main(["--camera-index", "-1"])
    assert exc.value.code == 2
