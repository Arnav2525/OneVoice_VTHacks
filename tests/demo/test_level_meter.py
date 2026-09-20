

from __future__ import annotations

import array
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from demo.level_meter import LevelMeterSink  # noqa: E402
from onevoice.core.models.audio_chunk import AudioChunk  # noqa: E402

def _chunk(samples: list[float], sample_rate: int = 16_000) -> AudioChunk:
    return AudioChunk(
        timestamp_ms=0.0,
        data=array.array("f", samples),
        sample_rate=sample_rate,
        channels=1,
        metadata={},
    )

def test_starts_at_zero() -> None:
    sink = LevelMeterSink()
    sink.start()
    rms, peak = sink.snapshot()
    assert rms == 0.0
    assert peak == 0.0

def test_silence_stays_at_zero() -> None:
    sink = LevelMeterSink()
    sink.start()
    sink.write(_chunk([0.0] * 320))
    rms, peak = sink.snapshot()
    assert rms == 0.0
    assert peak == 0.0

def test_loud_chunk_raises_both_rms_and_peak() -> None:
    sink = LevelMeterSink(rms_smoothing_alpha=1.0)
    sink.start()
    sink.write(_chunk([0.8] * 320))
    rms, peak = sink.snapshot()
    assert rms == pytest.approx(0.8)
    assert peak == pytest.approx(0.8)

def test_rms_smoothing_eases_toward_new_level_not_snaps() -> None:
    sink = LevelMeterSink(rms_smoothing_alpha=0.5)
    sink.start()
    sink.write(_chunk([0.0] * 320))
    sink.write(_chunk([1.0] * 320))
    rms, _ = sink.snapshot()

    assert 0.4 < rms < 0.6

def test_peak_snaps_up_instantly_on_a_louder_chunk() -> None:
    sink = LevelMeterSink(rms_smoothing_alpha=0.3, peak_decay_s=10.0)
    sink.start()
    sink.write(_chunk([0.2] * 320))
    _, peak_after_quiet = sink.snapshot()
    sink.write(_chunk([0.9] * 320))
    _, peak_after_loud = sink.snapshot()
    assert peak_after_loud > peak_after_quiet
    assert peak_after_loud == pytest.approx(0.9)

def test_peak_decays_over_wall_clock_time_when_nothing_louder_arrives() -> None:

    sink = LevelMeterSink(rms_smoothing_alpha=1.0, peak_decay_s=0.05)
    sink.start()
    sink.write(_chunk([1.0] * 320))
    _, peak_immediately = sink.snapshot()
    assert peak_immediately == pytest.approx(1.0)

    time.sleep(0.1)
    sink.write(_chunk([0.0] * 320))
    _, peak_after_decay = sink.snapshot()
    assert peak_after_decay < peak_immediately
    assert peak_after_decay == pytest.approx(0.0, abs=0.05)

def test_snapshot_is_always_clamped_to_zero_one() -> None:
    sink = LevelMeterSink()
    sink.start()

    sink.write(_chunk([5.0] * 320))
    rms, peak = sink.snapshot()
    assert 0.0 <= rms <= 1.0
    assert 0.0 <= peak <= 1.0

def test_empty_chunk_does_not_crash_or_change_state() -> None:
    sink = LevelMeterSink(rms_smoothing_alpha=1.0)
    sink.start()
    sink.write(_chunk([0.5] * 320))
    rms_before, peak_before = sink.snapshot()
    sink.write(_chunk([]))
    rms_after, peak_after = sink.snapshot()
    assert rms_after == rms_before
    assert peak_after == peak_before

def test_start_resets_state_between_runs() -> None:
    sink = LevelMeterSink(rms_smoothing_alpha=1.0, peak_decay_s=10.0)
    sink.start()
    sink.write(_chunk([0.9] * 320))
    rms, peak = sink.snapshot()
    assert rms > 0.0 and peak > 0.0

    sink.start()
    rms, peak = sink.snapshot()
    assert rms == 0.0
    assert peak == 0.0
