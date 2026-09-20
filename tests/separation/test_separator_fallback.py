

import array

import pytest

from onevoice.core.models.audio_chunk import AudioChunk
from onevoice.core.models.target_selection import TargetSelection
from onevoice.separation.adapters.base import AdapterContext, SeparatorAdapter
from onevoice.separation.base import LoadableSeparator
from onevoice.separation.config import SeparationConfig
from onevoice.separation.utilities import make_output_chunk

class EchoAdapter(SeparatorAdapter):
    def to_backend(self, model, audio_chunk, target, all_tracks, ctx):
        return audio_chunk.data

    def infer(self, model, backend_input, ctx):
        return backend_input

    def from_backend(self, backend_output, reference, ctx):
        return make_output_chunk(backend_output, reference, {"echo": True})

class RaisingAdapter(EchoAdapter):
    def infer(self, model, backend_input, ctx):
        raise RuntimeError("boom")

class FakeSeparator(LoadableSeparator):
    def __init__(self, config, adapter):
        super().__init__(config, adapter)

    def _build_model(self, config, device):
        return object()

def _chunk():
    return AudioChunk(
        timestamp_ms=1.0,
        data=array.array("f", [0.1, 0.2, 0.3]),
        sample_rate=16000,
        channels=1,
        metadata={"seq": 7},
    )

def _target():
    return TargetSelection(timestamp_ms=1.0, selected_speaker=None)

def test_successful_separation_lazy_loads():
    sep = FakeSeparator(SeparationConfig(device="cpu"), EchoAdapter())
    assert not sep.is_loaded
    out = sep.separate(_chunk(), _target(), [])
    assert sep.is_loaded
    assert out.metadata["echo"] is True
    assert out.sample_rate == 16000
    stats = sep.get_stats()
    assert stats["inferences"] == 1
    assert stats["failures"] == 0

def test_failure_falls_back_to_passthrough():
    sep = FakeSeparator(SeparationConfig(device="cpu"), RaisingAdapter())
    out = sep.separate(_chunk(), _target(), [])
    assert out.metadata["fallback"] is True
    assert out.data == _chunk().data
    assert sep.get_stats()["fallbacks"] == 1

def test_failure_reraises_when_fallback_disabled():

    cfg = SeparationConfig(device="cpu", fallback_on_error=False, warmup=False)
    sep = FakeSeparator(cfg, RaisingAdapter())
    with pytest.raises(RuntimeError):
        sep.separate(_chunk(), _target(), [])

def test_fallback_logged_only_once(caplog):
    sep = FakeSeparator(SeparationConfig(device="cpu"), RaisingAdapter())
    for _ in range(5):
        sep.separate(_chunk(), _target(), [])
    errors = [r for r in caplog.records if r.levelname == "ERROR"]
    assert len(errors) <= 1
    assert sep.get_stats()["fallbacks"] == 5

def test_context_dataclass_is_frozen():
    ctx = AdapterContext(device="cpu", config=SeparationConfig())
    with pytest.raises(Exception):
        ctx.device = "cuda"  # type: ignore[misc]
