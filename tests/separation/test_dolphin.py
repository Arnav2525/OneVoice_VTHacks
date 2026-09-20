

from __future__ import annotations

import array
import time

import pytest

from onevoice.core.models.audio_chunk import AudioChunk
from onevoice.core.models.speaker_track import SpeakerTrack
from onevoice.core.models.target_selection import TargetSelection
from onevoice.separation.config import SeparationConfig
from onevoice.separation.registry import available_backends, create_separator
from onevoice.separation.utilities import chunk_to_float_list

def _chunk(samples: int = 320, value: float = 0.1, ts: float = 0.0) -> AudioChunk:
    data = array.array("f", [value] * samples)
    return AudioChunk(
        timestamp_ms=ts, data=data, sample_rate=16000, channels=1, metadata={}
    )

def _speaker(tid: str, patch: list[float] | None = None) -> SpeakerTrack:
    meta: dict = {}
    if patch is not None:
        meta["lip_patch"] = patch
    return SpeakerTrack(tid, (0, 0, 10, 10), 0.9, metadata=meta)

def test_registry_lists_dolphin():
    assert "dolphin" in available_backends()

def test_no_target_returns_silence():

    sep = create_separator({"name": "dolphin", "device": "cpu", "warmup": False})
    out = sep.separate(_chunk(), TargetSelection(0.0, None), [])
    assert out.metadata.get("no_target") is True
    assert max(abs(x) for x in chunk_to_float_list(out.data)) == 0.0

def test_no_target_passthrough_policy():
    sep = create_separator(
        {
            "name": "dolphin",
            "device": "cpu",
            "warmup": False,
            "params": {"no_target_policy": "passthrough"},
        }
    )
    chunk = _chunk(value=0.42)
    out = sep.separate(chunk, TargetSelection(0.0, None), [])
    assert out.metadata.get("no_target") is True
    assert chunk_to_float_list(out.data) == chunk_to_float_list(chunk.data)

def test_precision_alias_fp16():
    cfg = SeparationConfig(name="dolphin", precision="fp16")
    assert cfg.precision == "float16"

def _stub_model(mix, mouth):
    return mix

@pytest.fixture
def dolphin_adapter():
    torch = pytest.importorskip("torch")
    from onevoice.separation.adapters.dolphin_adapter import DolphinAdapter

    config = SeparationConfig(
        name="dolphin",
        device="cpu",
        sample_rate=16000,
        chunk_size=320,
        params={"window_s": 0.1, "hop_s": 0.05},
    )
    return DolphinAdapter(config), torch

def test_silence_before_window_full(dolphin_adapter):
    adapter, _ = dolphin_adapter
    from onevoice.separation.adapters.base import AdapterContext

    config = SeparationConfig(name="dolphin", device="cpu", sample_rate=16000)
    ctx = AdapterContext(device="cpu", config=config)
    target = TargetSelection(0.0, _speaker("spk-1", [0.1] * 64))

    chunk = _chunk(samples=320, value=0.5)
    backend_input = adapter.to_backend(_stub_model, chunk, target, [], ctx)
    out = adapter.infer(_stub_model, backend_input, ctx)
    result = adapter.from_backend(out, chunk, ctx)

    assert max(abs(x) for x in chunk_to_float_list(result.data)) == 0.0
    assert result.metadata["conditioning"] == "buffering"

def test_real_output_after_window_fills(dolphin_adapter):
    adapter, _ = dolphin_adapter
    from onevoice.separation.adapters.base import AdapterContext

    config = SeparationConfig(name="dolphin", device="cpu", sample_rate=16000)
    ctx = AdapterContext(device="cpu", config=config)
    target = TargetSelection(0.0, _speaker("spk-1", [0.3] * 64))

    for i in range(5):
        chunk = _chunk(samples=320, value=0.5, ts=float(i * 20))
        backend_input = adapter.to_backend(_stub_model, chunk, target, [], ctx)
        adapter.infer(_stub_model, backend_input, ctx)

    adapter.flush_pending("spk-1")

    chunk = _chunk(samples=320, value=0.5, ts=100.0)
    backend_input = adapter.to_backend(_stub_model, chunk, target, [], ctx)
    out = adapter.infer(_stub_model, backend_input, ctx)
    last_result = adapter.from_backend(out, chunk, ctx)

    assert max(abs(x) for x in chunk_to_float_list(last_result.data)) > 0.0
    assert last_result.metadata["conditioning"] in {"visual", "audio_only"}
    assert last_result.metadata["target_track_id"] == "spk-1"

def test_state_resets_on_track_switch(dolphin_adapter):
    adapter, _ = dolphin_adapter
    from onevoice.separation.adapters.base import AdapterContext

    config = SeparationConfig(name="dolphin", device="cpu", sample_rate=16000)
    ctx = AdapterContext(device="cpu", config=config)
    t1 = TargetSelection(0.0, _speaker("a", [0.1] * 64))
    t2 = TargetSelection(100.0, _speaker("b", [0.2] * 64))

    chunk = _chunk()
    backend_input = adapter.to_backend(_stub_model, chunk, t1, [], ctx)
    adapter.infer(_stub_model, backend_input, ctx)
    assert "a" in adapter._track_states  # noqa: SLF001

    backend_input = adapter.to_backend(_stub_model, chunk, t2, [], ctx)
    adapter.infer(_stub_model, backend_input, ctx)
    assert "a" not in adapter._track_states  # noqa: SLF001
    assert "b" in adapter._track_states  # noqa: SLF001

def _slow_stub_model(mix, mouth):
    time.sleep(0.2)
    return mix

def test_track_switch_warns_on_discarded_in_flight_window(dolphin_adapter, caplog):
    adapter, _ = dolphin_adapter
    from onevoice.separation.adapters.base import AdapterContext

    config = SeparationConfig(name="dolphin", device="cpu", sample_rate=16000)
    ctx = AdapterContext(device="cpu", config=config)
    t1 = TargetSelection(0.0, _speaker("a", [0.1] * 64))
    t2 = TargetSelection(100.0, _speaker("b", [0.2] * 64))

    for i in range(5):
        chunk = _chunk(samples=320, ts=float(i * 20))
        backend_input = adapter.to_backend(_slow_stub_model, chunk, t1, [], ctx)
        adapter.infer(_slow_stub_model, backend_input, ctx)
    assert adapter._track_states["a"].pending_future is not None  # noqa: SLF001

    with caplog.at_level("WARNING"):
        chunk = _chunk(samples=320, ts=100.0)
        backend_input = adapter.to_backend(_slow_stub_model, chunk, t2, [], ctx)
        adapter.infer(_slow_stub_model, backend_input, ctx)

    assert any("IN-FLIGHT" in r.message and "track a" in r.message for r in caplog.records)
    adapter.shutdown()

def test_track_switch_warns_on_discarded_completed_window(dolphin_adapter, caplog):
    adapter, _ = dolphin_adapter
    from onevoice.separation.adapters.base import AdapterContext

    config = SeparationConfig(name="dolphin", device="cpu", sample_rate=16000)
    ctx = AdapterContext(device="cpu", config=config)
    t1 = TargetSelection(0.0, _speaker("a", [0.1] * 64))
    t2 = TargetSelection(100.0, _speaker("b", [0.2] * 64))

    for i in range(5):
        chunk = _chunk(samples=320, ts=float(i * 20))
        backend_input = adapter.to_backend(_stub_model, chunk, t1, [], ctx)
        adapter.infer(_stub_model, backend_input, ctx)

    adapter._track_states["a"].pending_future.result(timeout=5.0)  # noqa: SLF001

    with caplog.at_level("WARNING"):
        chunk = _chunk(samples=320, ts=100.0)
        backend_input = adapter.to_backend(_stub_model, chunk, t2, [], ctx)
        adapter.infer(_stub_model, backend_input, ctx)

    assert any("COMPLETED" in r.message and "track a" in r.message for r in caplog.records)

def test_drain_and_warn_on_shutdown_times_out_on_slow_window(dolphin_adapter, caplog):
    adapter, _ = dolphin_adapter
    from onevoice.separation.adapters.base import AdapterContext

    config = SeparationConfig(name="dolphin", device="cpu", sample_rate=16000)
    ctx = AdapterContext(device="cpu", config=config)
    target = TargetSelection(0.0, _speaker("spk-1", [0.1] * 64))

    for i in range(5):
        chunk = _chunk(samples=320, ts=float(i * 20))
        backend_input = adapter.to_backend(_slow_stub_model, chunk, target, [], ctx)
        adapter.infer(_slow_stub_model, backend_input, ctx)
    assert adapter._track_states["spk-1"].pending_future is not None  # noqa: SLF001

    with caplog.at_level("WARNING"):
        adapter.drain_and_warn_on_shutdown(timeout_s=0.01)

    assert any(
        "in-flight window still computing" in r.message and "spk-1" in r.message
        for r in caplog.records
    )
    adapter.shutdown()

def test_drain_and_warn_on_shutdown_collects_completed_window(dolphin_adapter, caplog):
    adapter, _ = dolphin_adapter
    from onevoice.separation.adapters.base import AdapterContext

    config = SeparationConfig(name="dolphin", device="cpu", sample_rate=16000)
    ctx = AdapterContext(device="cpu", config=config)
    target = TargetSelection(0.0, _speaker("spk-1", [0.1] * 64))

    for i in range(5):
        chunk = _chunk(samples=320, ts=float(i * 20))
        backend_input = adapter.to_backend(_stub_model, chunk, target, [], ctx)
        adapter.infer(_stub_model, backend_input, ctx)

    with caplog.at_level("WARNING"):
        adapter.drain_and_warn_on_shutdown(timeout_s=5.0)

    assert any(
        "collected a completed window" in r.message and "spk-1" in r.message
        for r in caplog.records
    )

def _quiet_stub_model(mix, mouth):
    return mix * 0.01

def test_auto_gain_boosts_quiet_output_to_match_input(dolphin_adapter):
    _, torch = dolphin_adapter
    from onevoice.separation.adapters.base import AdapterContext
    from onevoice.separation.adapters.dolphin_adapter import DolphinAdapter

    config = SeparationConfig(
        name="dolphin", device="cpu", sample_rate=16000, chunk_size=320,
        params={"window_s": 0.1, "hop_s": 0.05},
    )
    adapter = DolphinAdapter(config)
    ctx = AdapterContext(device="cpu", config=config)
    target = TargetSelection(0.0, _speaker("spk-1", [0.3] * 64))

    for i in range(5):
        chunk = _chunk(samples=320, value=0.5, ts=float(i * 20))
        backend_input = adapter.to_backend(_quiet_stub_model, chunk, target, [], ctx)
        adapter.infer(_quiet_stub_model, backend_input, ctx)
    adapter.flush_pending("spk-1")

    chunk = _chunk(samples=320, value=0.5, ts=100.0)
    backend_input = adapter.to_backend(_quiet_stub_model, chunk, target, [], ctx)
    out = adapter.infer(_quiet_stub_model, backend_input, ctx)

    assert max(abs(x) for x in out) >= 0.4
    adapter.shutdown()

def test_auto_gain_disabled_leaves_output_quiet(dolphin_adapter):
    _, torch = dolphin_adapter
    from onevoice.separation.adapters.base import AdapterContext
    from onevoice.separation.adapters.dolphin_adapter import DolphinAdapter

    config = SeparationConfig(
        name="dolphin", device="cpu", sample_rate=16000, chunk_size=320,
        params={"window_s": 0.1, "hop_s": 0.05, "auto_gain": False},
    )
    adapter = DolphinAdapter(config)
    ctx = AdapterContext(device="cpu", config=config)
    target = TargetSelection(0.0, _speaker("spk-1", [0.3] * 64))

    for i in range(5):
        chunk = _chunk(samples=320, value=0.5, ts=float(i * 20))
        backend_input = adapter.to_backend(_quiet_stub_model, chunk, target, [], ctx)
        adapter.infer(_quiet_stub_model, backend_input, ctx)
    adapter.flush_pending("spk-1")

    chunk = _chunk(samples=320, value=0.5, ts=100.0)
    backend_input = adapter.to_backend(_quiet_stub_model, chunk, target, [], ctx)
    out = adapter.infer(_quiet_stub_model, backend_input, ctx)

    assert max(abs(x) for x in out) < 0.05
    adapter.shutdown()

def _true_silence_stub_model(mix, mouth):
    return mix * 0.0

def test_auto_gain_skips_true_silence_without_nan(dolphin_adapter):
    _, torch = dolphin_adapter
    from onevoice.separation.adapters.base import AdapterContext
    from onevoice.separation.adapters.dolphin_adapter import DolphinAdapter

    config = SeparationConfig(
        name="dolphin", device="cpu", sample_rate=16000, chunk_size=320,
        params={"window_s": 0.1, "hop_s": 0.05},
    )
    adapter = DolphinAdapter(config)
    ctx = AdapterContext(device="cpu", config=config)
    target = TargetSelection(0.0, _speaker("spk-1", [0.3] * 64))

    for i in range(5):
        chunk = _chunk(samples=320, value=0.5, ts=float(i * 20))
        backend_input = adapter.to_backend(_true_silence_stub_model, chunk, target, [], ctx)
        adapter.infer(_true_silence_stub_model, backend_input, ctx)
    adapter.flush_pending("spk-1")

    chunk = _chunk(samples=320, value=0.5, ts=100.0)
    backend_input = adapter.to_backend(_true_silence_stub_model, chunk, target, [], ctx)
    out = adapter.infer(_true_silence_stub_model, backend_input, ctx)

    import math

    result = list(out)
    assert all(math.isfinite(x) for x in result)
    assert max(abs(x) for x in result) == 0.0
    adapter.shutdown()

def _below_noise_floor_stub_model(mix, mouth):
    return mix * 0.00001

def test_auto_gain_leaves_below_noise_floor_window_unboosted(dolphin_adapter):
    _, torch = dolphin_adapter
    from onevoice.separation.adapters.base import AdapterContext
    from onevoice.separation.adapters.dolphin_adapter import DolphinAdapter

    config = SeparationConfig(
        name="dolphin", device="cpu", sample_rate=16000, chunk_size=320,
        params={"window_s": 0.1, "hop_s": 0.05},
    )
    adapter = DolphinAdapter(config)
    ctx = AdapterContext(device="cpu", config=config)
    target = TargetSelection(0.0, _speaker("spk-1", [0.3] * 64))

    for i in range(5):
        chunk = _chunk(samples=320, value=0.5, ts=float(i * 20))
        backend_input = adapter.to_backend(_below_noise_floor_stub_model, chunk, target, [], ctx)
        adapter.infer(_below_noise_floor_stub_model, backend_input, ctx)
    adapter.flush_pending("spk-1")

    chunk = _chunk(samples=320, value=0.5, ts=100.0)
    backend_input = adapter.to_backend(_below_noise_floor_stub_model, chunk, target, [], ctx)
    out = adapter.infer(_below_noise_floor_stub_model, backend_input, ctx)

    assert max(abs(x) for x in out) < 0.001
    adapter.shutdown()

def test_patch_to_frame_uses_real_2d_crop_directly(dolphin_adapter):
    adapter, _ = dolphin_adapter
    import numpy as np

    size = adapter._mouth_size
    real_crop = np.arange(size * size, dtype=np.float32).reshape(size, size)
    frame = adapter._patch_to_frame(real_crop)
    assert frame.shape == (size, size)

    assert np.array_equal(frame, real_crop)

def test_patch_to_frame_resizes_mismatched_2d_crop(dolphin_adapter):
    adapter, _ = dolphin_adapter
    import numpy as np

    size = adapter._mouth_size
    small_crop = np.arange(20 * 10, dtype=np.float32).reshape(20, 10)
    frame = adapter._patch_to_frame(small_crop)
    assert frame.shape == (size, size)

def test_patch_to_frame_still_pads_flat_legacy_patch(dolphin_adapter):
    adapter, _ = dolphin_adapter
    import numpy as np

    size = adapter._mouth_size
    flat_patch = [0.3] * 64
    frame = adapter._patch_to_frame(flat_patch)
    assert frame.shape == (size, size)
    assert np.count_nonzero(frame) == 64

def test_retinaface_path_uses_mock_aligner(dolphin_adapter):

    adapter, _ = dolphin_adapter
    import numpy as np
    from onevoice.separation.adapters.base import AdapterContext
    from onevoice.separation.config import SeparationConfig

    adapter._visual_aligner = "retinaface"  # noqa: SLF001
    calls: list[int] = []

    class _FakeAligner:
        def align_window(self, frames_bgr, window_margin=12):
            calls.append(len(frames_bgr))
            t = len(frames_bgr)

            return np.full((t, adapter._mouth_size, adapter._mouth_size), 0.4, dtype=np.float32)

    adapter._aligner = _FakeAligner()  # noqa: SLF001

    config = SeparationConfig(name="dolphin", device="cpu", sample_rate=16000)
    ctx = AdapterContext(device="cpu", config=config)

    def _spk(tid: str) -> SpeakerTrack:
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        return SpeakerTrack(
            tid,
            (0, 0, 10, 10),
            0.9,
            metadata={"lip_patch": [0.1] * 64, "frame_bgr": frame},
        )

    target = TargetSelection(0.0, _spk("spk-1"))
    for i in range(5):
        chunk = _chunk(samples=320, value=0.5, ts=float(i * 20))
        backend_input = adapter.to_backend(_stub_model, chunk, target, [], ctx)
        adapter.infer(_stub_model, backend_input, ctx)

    adapter.flush_pending("spk-1")
    assert calls, "align_window should have been invoked on the retinaface path"
    assert calls[0] > 0

def test_output_length_matches_input_every_call(dolphin_adapter):
    adapter, _ = dolphin_adapter
    from onevoice.separation.adapters.base import AdapterContext

    config = SeparationConfig(name="dolphin", device="cpu", sample_rate=16000)
    ctx = AdapterContext(device="cpu", config=config)
    target = TargetSelection(0.0, _speaker("spk-1", [0.3] * 64))

    for i in range(10):
        chunk = _chunk(samples=320, ts=float(i * 20))
        backend_input = adapter.to_backend(_stub_model, chunk, target, [], ctx)
        out = adapter.infer(_stub_model, backend_input, ctx)
        result = adapter.from_backend(out, chunk, ctx)
        assert len(chunk_to_float_list(result.data)) == 320

def _dolphin_available() -> bool:
    try:
        from onevoice.separation.dolphin_loader import vendor_root

        vendor_root()
        return True
    except Exception:
        return False

@pytest.mark.skipif(not _dolphin_available(), reason="Dolphin vendor tree missing (scripts/vendor_dolphin.py)")
def test_real_dolphin_loads_and_separates():
    pytest.importorskip("torch")
    sep = create_separator(
        {
            "name": "dolphin",
            "device": "cpu",
            "warmup": False,
            "params": {"window_s": 0.2, "hop_s": 0.1},
        }
    )
    try:
        sep.load()
    except Exception as exc:  # noqa: BLE001 - no network / no cached HF weights
        pytest.skip(f"Dolphin weights unavailable: {exc}")

    target = TargetSelection(0.0, _speaker("spk-1", [0.3] * 64))
    out = None
    for i in range(20):
        out = sep.separate(_chunk(samples=320, ts=float(i * 20)), target, [target.selected_speaker])  # type: ignore[list-item]

        sep._adapter_impl.flush_pending("spk-1")  # noqa: SLF001
    sep._adapter_impl.flush_pending("spk-1")  # noqa: SLF001 - drain any in-flight window (async pipelining)
    out = sep.separate(_chunk(samples=320, ts=400.0), target, [target.selected_speaker])  # type: ignore[list-item]
    assert out is not None
    assert max(abs(x) for x in chunk_to_float_list(out.data)) > 0.0
    status = sep.get_status()
    assert status["backend"] == "dolphin"
