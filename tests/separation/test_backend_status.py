

import pytest

from onevoice.separation.config import SeparationConfig
from onevoice.separation.registry import create_separator
from onevoice.separation.utilities import empty_target, silent_chunk

def _sep(**overrides):
    config = {"name": "asteroid", "device": "cpu", **overrides}
    return create_separator(config)

def test_status_before_use_is_unloaded_and_unhealthy():
    sep = _sep()
    status = sep.get_status()
    assert status["backend"] == "asteroid"
    assert status["loaded"] is False
    assert status["healthy"] is False
    assert status["fallback_active"] is False
    assert status["failures"] == 0

def test_fallback_is_explicit_and_marked(caplog):
    sep = _sep()
    chunk = silent_chunk(16000, 320)
    with caplog.at_level("ERROR"):
        out = sep.separate(chunk, empty_target(), [])

    assert out.metadata["fallback"] is True
    assert out.metadata["backend"] == "asteroid"
    assert "fallback_reason" in out.metadata
    assert list(out.data) == list(chunk.data)

    status = sep.get_status()
    assert status["fallback_active"] is True
    assert status["healthy"] is False
    assert status["failures"] == 1
    assert status["last_error"] is not None

    assert any(rec.levelname == "ERROR" for rec in caplog.records)

def test_first_degrade_logged_once_then_quiet(caplog):
    sep = _sep()
    chunk = silent_chunk(16000, 320)
    with caplog.at_level("ERROR"):
        for _ in range(5):
            sep.separate(chunk, empty_target(), [])
    errors = [r for r in caplog.records if r.levelname == "ERROR"]
    assert len(errors) == 1
    assert sep.get_status()["fallbacks"] == 5

def test_fallback_disabled_reraises():
    sep = _sep(fallback_on_error=False)
    chunk = silent_chunk(16000, 320)
    with pytest.raises(Exception):
        sep.separate(chunk, empty_target(), [])

def test_config_from_raw_roundtrip():
    config = SeparationConfig.from_raw({"name": "asteroid", "device": "cpu"})
    assert config.name == "asteroid"
    assert config.device == "cpu"

def test_passthrough_status_is_never_real_separation():
    sep = create_separator("passthrough")
    status = sep.get_status()
    assert status["backend"] == "passthrough"
    assert status["is_real_separation"] is False
    assert status["healthy"] is True

def test_loadable_status_not_real_until_loaded():
    sep = _sep()
    assert sep.get_status()["is_real_separation"] is False
    sep.separate(silent_chunk(16000, 320), empty_target(), [])

    assert sep.get_status()["is_real_separation"] is False
