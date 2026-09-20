import time

import pytest

from onevoice.separation.config import SeparationConfig
from onevoice.separation.exceptions import BackendInitializationError
from onevoice.separation.loader import BackendLoader

def _cfg(**kw):
    return SeparationConfig(device="cpu", warmup=False, **kw)

def test_lazy_load():
    calls = []

    def builder(config, device):
        calls.append(device)
        return {"model": True}

    loader = BackendLoader(_cfg(), builder)
    assert not loader.is_loaded
    assert calls == []
    model = loader.get()
    assert model == {"model": True}
    assert loader.is_loaded
    loader.get()
    assert len(calls) == 1

def test_warmup_runs():
    warmups = []

    def builder(config, device):
        return object()

    def warmup(model, config, device):
        warmups.append(device)

    cfg = SeparationConfig(device="cpu", warmup=True, warmup_iterations=3)
    BackendLoader(cfg, builder, warmup).load()
    assert len(warmups) == 3

def test_checkpoint_missing_maps_to_init_error():
    def builder(config, device):
        raise FileNotFoundError("no ckpt")

    with pytest.raises(BackendInitializationError):
        BackendLoader(_cfg(), builder).load()

def test_generic_failure_maps_to_init_error():
    def builder(config, device):
        raise RuntimeError("kaboom")

    with pytest.raises(BackendInitializationError):
        BackendLoader(_cfg(), builder).load()

def test_warmup_failure_unloads_model():
    def builder(config, device):
        return object()

    def warmup(model, config, device):
        raise RuntimeError("bad warmup")

    cfg = SeparationConfig(device="cpu", warmup=True)
    loader = BackendLoader(cfg, builder, warmup)
    with pytest.raises(BackendInitializationError):
        loader.load()
    assert not loader.is_loaded

def test_second_load_after_failure_raises():
    def builder(config, device):
        raise RuntimeError("kaboom")

    loader = BackendLoader(_cfg(), builder)
    with pytest.raises(BackendInitializationError):
        loader.load()
    with pytest.raises(BackendInitializationError):
        loader.load()

def test_unload_resets():
    loader = BackendLoader(_cfg(), lambda c, d: object())
    loader.load()
    assert loader.is_loaded
    loader.unload()
    assert not loader.is_loaded
    assert loader.load_time_ms == 0.0
    assert loader.warmup_time_ms == 0.0

def test_load_and_warmup_time_recorded():
    def builder(config, device):
        time.sleep(0.02)
        return object()

    def warmup(model, config, device):
        time.sleep(0.01)

    cfg = SeparationConfig(device="cpu", warmup=True, warmup_iterations=1)
    loader = BackendLoader(cfg, builder, warmup)
    loader.load()
    assert loader.load_time_ms >= 15.0
    assert loader.warmup_time_ms >= 5.0

def test_no_warmup_leaves_warmup_time_zero():
    loader = BackendLoader(_cfg(), lambda c, d: object())
    loader.load()
    assert loader.warmup_time_ms == 0.0
