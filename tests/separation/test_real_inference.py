

import array

import pytest

torch = pytest.importorskip("torch")

from onevoice.core.models.audio_chunk import AudioChunk
from onevoice.core.models.target_selection import TargetSelection
from onevoice.separation.adapters.base import SeparatorAdapter
from onevoice.separation.base import LoadableSeparator
from onevoice.separation.config import SeparationConfig

class _GainModel(torch.nn.Module):

    def __init__(self, gain: float = 0.5) -> None:
        super().__init__()
        self.gain = gain

    def forward(self, x):  # type: ignore[no-untyped-def]
        return x * self.gain

class _GainAdapter(SeparatorAdapter):
    def infer(self, model, backend_input, ctx):
        with torch.no_grad():
            return model(backend_input)

class _GainSeparator(LoadableSeparator):
    def __init__(self, config: SeparationConfig) -> None:
        super().__init__(config, _GainAdapter())

    def _build_model(self, config: SeparationConfig, device: str):
        return _GainModel().to(device).eval()

def _tone(n: int, sr: int = 16000) -> array.array:
    import math

    step = 2 * math.pi * 220 / sr
    return array.array("f", [0.5 * math.sin(step * i) for i in range(n)])

def _chunk(data: array.array, sr: int = 16000) -> AudioChunk:
    return AudioChunk(
        timestamp_ms=0.0, data=data, sample_rate=sr, channels=1, metadata={}
    )

def _target() -> TargetSelection:
    return TargetSelection(timestamp_ms=0.0, selected_speaker=None)

def test_real_inference_changes_audio_and_preserves_contract():
    cfg = SeparationConfig(name="gain", device="cpu", sample_rate=16000, warmup=True)
    sep = _GainSeparator(cfg)
    src = _chunk(_tone(1600))
    out = sep.separate(src, _target(), [])

    assert out.metadata.get("fallback") is None
    assert out.sample_rate == 16000
    assert len(out.data) == len(src.data)

    assert out.data[100] == pytest.approx(src.data[100] * 0.5, rel=1e-3)
    assert max(abs(x) for x in out.data) <= 1.0

    status = sep.get_status()
    assert status["is_real_separation"] is True
    assert status["healthy"] is True
    assert status["load_time_ms"] >= 0.0
    assert status["inferences"] == 1

def test_resample_roundtrip_preserves_length_and_rate():

    cfg = SeparationConfig(name="gain", device="cpu", sample_rate=8000, resample=True)
    sep = _GainSeparator(cfg)
    src = _chunk(_tone(1600, sr=16000), sr=16000)
    out = sep.separate(src, _target(), [])

    assert out.sample_rate == 16000
    assert len(out.data) == len(src.data)
    assert sep.get_status()["is_real_separation"] is True

def test_unsupported_sample_rate_without_resample_falls_back():
    cfg = SeparationConfig(
        name="gain", device="cpu", sample_rate=8000, resample=False
    )
    sep = _GainSeparator(cfg)
    src = _chunk(_tone(800, sr=16000), sr=16000)
    out = sep.separate(src, _target(), [])

    assert out.metadata["fallback"] is True
    assert sep.get_status()["is_real_separation"] is False
