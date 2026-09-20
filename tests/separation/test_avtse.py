

from __future__ import annotations

import array

import pytest

from onevoice.core.models.audio_chunk import AudioChunk
from onevoice.core.models.speaker_track import SpeakerTrack
from onevoice.core.models.target_selection import TargetSelection
from onevoice.separation.config import SeparationConfig
from onevoice.separation.registry import create_separator
from onevoice.separation.utilities import chunk_to_float_list

def _chunk(samples: int = 320) -> AudioChunk:
    data = array.array("f", [0.1] * samples)
    return AudioChunk(
        timestamp_ms=100.0,
        data=data,
        sample_rate=16000,
        channels=1,
        metadata={},
    )

def _speaker(tid: str, patch: list[float] | None = None) -> SpeakerTrack:
    meta: dict = {"lip_roi_ts_ms": 100.0}
    if patch is not None:
        meta["lip_patch"] = patch
    return SpeakerTrack(tid, (0, 0, 10, 10), 0.9, metadata=meta)

@pytest.fixture
def avtse():
    pytest.importorskip("torch")
    return create_separator(
        {
            "name": "avtse",
            "device": "cpu",
            "warmup": True,
            "params": {"no_target_policy": "silence"},
        }
    )

def test_registry_loads_avtse():
    pytest.importorskip("torch")
    sep = create_separator({"name": "avtse", "device": "cpu", "warmup": False})
    sep.load()
    status = sep.get_status()
    assert status["backend"] == "avtse"
    assert status["is_real_separation"] is True

def test_no_target_returns_silence(avtse):
    out = avtse.separate(_chunk(), TargetSelection(100.0, None), [])
    assert out.metadata.get("no_target") is True
    assert max(abs(x) for x in chunk_to_float_list(out.data)) == 0.0

def test_real_separation_with_target(avtse):
    patch = [0.0] * 64
    target = TargetSelection(100.0, _speaker("spk-1", patch))
    out = avtse.separate(_chunk(), target, [target.selected_speaker])  # type: ignore[list-item]
    assert out.metadata.get("conditioning") in {"visual", "audio_only", "stale_visual"}
    assert out.metadata.get("target_track_id") == "spk-1"
    status = avtse.get_status()
    assert status["is_real_separation"] is True
    assert status["inferences"] >= 1

def test_streaming_state_resets_on_switch(avtse):
    adapter = avtse._adapter_impl  # noqa: SLF001
    t1 = _speaker("a", [0.1] * 64)
    t2 = _speaker("b", [0.2] * 64)
    avtse.separate(_chunk(), TargetSelection(100.0, t1), [t1])
    assert "a" in adapter._track_states  # noqa: SLF001
    avtse.separate(_chunk(), TargetSelection(200.0, t2), [t2])
    assert "a" not in adapter._track_states  # noqa: SLF001
    assert "b" in adapter._track_states  # noqa: SLF001

def test_precision_alias_fp16():
    cfg = SeparationConfig(name="avtse", precision="fp16")
    assert cfg.precision == "float16"
