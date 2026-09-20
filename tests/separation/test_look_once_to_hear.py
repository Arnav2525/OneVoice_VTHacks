

from __future__ import annotations

import array
from pathlib import Path

import pytest

from onevoice.core.models.audio_chunk import AudioChunk
from onevoice.core.models.speaker_track import SpeakerTrack
from onevoice.core.models.target_selection import TargetSelection
from onevoice.separation.config import SeparationConfig
from onevoice.separation.loth.model_loader import (
    DEFAULT_LOTH_ROOT,
    resolve_checkpoint,
)
from onevoice.separation.registry import create_separator
from onevoice.separation.utilities import chunk_to_float_list

def _chunk(samples: int = 320, value: float = 0.1) -> AudioChunk:
    data = array.array("f", [value] * samples)
    return AudioChunk(
        timestamp_ms=100.0,
        data=data,
        sample_rate=16000,
        channels=1,
        metadata={},
    )

def _speaker(tid: str) -> SpeakerTrack:
    return SpeakerTrack(tid, (0, 0, 10, 10), 0.9, metadata={})

def _loth_available() -> bool:
    root = DEFAULT_LOTH_ROOT
    if not root.is_dir():
        return False
    try:
        from onevoice.separation.loth.model_loader import (
            DEFAULT_EMBED_CHECKPOINT,
            DEFAULT_TSH_CHECKPOINT,
        )

        return (
            (root / DEFAULT_TSH_CHECKPOINT).is_file()
            and (root / DEFAULT_EMBED_CHECKPOINT).is_file()
        )
    except Exception:
        return False

@pytest.fixture
def loth_config() -> dict:
    return {
        "name": "look_once_to_hear",
        "device": "cpu",
        "warmup": False,
        "params": {"no_target_policy": "silence", "enrollment_seconds": 0.05},
    }

def test_registry_lists_look_once_to_hear():
    from onevoice.separation.registry import available_backends

    assert "look_once_to_hear" in available_backends()

def test_no_target_returns_silence(loth_config):
    pytest.importorskip("torch")
    sep = create_separator(loth_config)
    if not _loth_available():
        pytest.skip("LookOnceToHear weights not installed (run scripts/setup_loth.py)")
    sep.load()
    out = sep.separate(_chunk(), TargetSelection(100.0, None), [])
    assert out.metadata.get("no_target") is True
    assert max(abs(x) for x in chunk_to_float_list(out.data)) == 0.0

def test_enrolling_passthrough_before_embed(loth_config):
    pytest.importorskip("torch")
    if not _loth_available():
        pytest.skip("LookOnceToHear weights not installed")
    sep = create_separator(loth_config)
    sep.load()
    target = TargetSelection(100.0, _speaker("spk-1"))
    out = sep.separate(_chunk(), target, [target.selected_speaker])  # type: ignore[list-item]
    assert out.metadata.get("enrolling") is True
    assert out.metadata.get("enrolled") is False
    assert out.metadata.get("conditioning") == "enrolling"

@pytest.mark.skipif(not _loth_available(), reason="LOTH weights missing")
def test_real_separation_after_enrollment():
    pytest.importorskip("torch")
    sep = create_separator(
        {
            "name": "look_once_to_hear",
            "device": "cpu",
            "warmup": True,
            "params": {"enrollment_seconds": 0.05, "no_target_policy": "silence"},
        }
    )
    sep.load()
    target = TargetSelection(100.0, _speaker("spk-1"))
    chunks = 30
    for i in range(chunks):
        out = sep.separate(
            _chunk(value=0.05 * (i % 5)),
            TargetSelection(100.0 + i, target.selected_speaker),
            [target.selected_speaker],  # type: ignore[list-item]
        )
    assert out.metadata.get("enrolled") is True
    assert out.metadata.get("conditioning") == "binaural_embed"
    status = sep.get_status()
    assert status["is_real_separation"] is True
    assert status["inferences"] >= chunks

def test_resolve_checkpoint_path(tmp_path: Path):
    ckpt = tmp_path / "best.ckpt"
    ckpt.write_bytes(b"fake")
    cfg = SeparationConfig(
        name="look_once_to_hear",
        checkpoint=str(ckpt),
    )
    assert resolve_checkpoint(
        cfg,
        tmp_path,
        rel_key="checkpoint_rel",
        default_rel="runs/tsh/best.ckpt",
    ) == ckpt.resolve()
