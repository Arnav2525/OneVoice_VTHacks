import pytest

from onevoice.separation.config import SeparationConfig

def test_defaults():
    cfg = SeparationConfig()
    assert cfg.name == "passthrough"
    assert cfg.device == "auto"
    assert cfg.precision == "float32"
    assert cfg.batch_size == 1

def test_from_raw_string():
    cfg = SeparationConfig.from_raw("RAVEN")
    assert cfg.name == "raven"

def test_from_raw_dict():
    cfg = SeparationConfig.from_raw({"name": "SpeechBrain", "device": "cpu"})
    assert cfg.name == "speechbrain"
    assert cfg.device == "cpu"

def test_from_raw_none():
    assert SeparationConfig.from_raw(None).name == "passthrough"

def test_invalid_precision():
    with pytest.raises(ValueError):
        SeparationConfig(precision="int8")

def test_invalid_device():
    with pytest.raises(ValueError):
        SeparationConfig(device="tpu")

def test_invalid_batch_size():
    with pytest.raises(ValueError):
        SeparationConfig(batch_size=0)

def test_cuda_indexed_device_ok():
    assert SeparationConfig(device="cuda:1").device == "cuda:1"

def test_extra_params_allowed():
    cfg = SeparationConfig.from_raw({"name": "asteroid", "custom_key": 5})
    assert cfg.custom_key == 5  # type: ignore[attr-defined]
