

import array

from onevoice.core.models.audio_chunk import AudioChunk
from onevoice.separation.passthrough import PassthroughSeparator
from onevoice.telemetry.audio_validation import (
    build_scenarios,
    run_validation,
    validate_output,
)

def _chunk(data, sr=16000, channels=1):
    return AudioChunk(
        timestamp_ms=0.0,
        data=array.array("f", data),
        sample_rate=sr,
        channels=channels,
        metadata={},
    )

def test_build_scenarios_covers_all_cases():
    names = {s.name for s in build_scenarios(16000, 1600)}
    assert names == {
        "single_speaker",
        "two_speakers",
        "multi_speaker",
        "background_noise",
    }

def test_validate_output_passes_for_faithful_copy():
    src = _chunk([0.1, 0.2, 0.3])
    out = _chunk([0.1, 0.2, 0.3])
    result = validate_output(src, out)
    assert result.ok
    assert result.length_ok
    assert result.sample_rate_ok
    assert result.clipping_ratio == 0.0

def test_validate_output_flags_wrong_length():
    src = _chunk([0.1, 0.2, 0.3])
    out = _chunk([0.1, 0.2])
    result = validate_output(src, out)
    assert not result.ok
    assert not result.length_ok

def test_validate_output_flags_wrong_sample_rate():
    src = _chunk([0.1, 0.2, 0.3], sr=16000)
    out = _chunk([0.1, 0.2, 0.3], sr=8000)
    result = validate_output(src, out)
    assert not result.ok
    assert not result.sample_rate_ok

def test_validate_output_flags_clipping():
    src = _chunk([0.0] * 100)
    out = _chunk([2.0] * 100)
    result = validate_output(src, out)
    assert not result.ok
    assert result.clipping_ratio == 1.0

def test_run_validation_passthrough_passes_contract():
    report = run_validation(
        PassthroughSeparator(), sample_rate=16000, chunk_samples=800
    )
    assert report.passed
    assert len(report.scenarios) == 4
    assert report.backend_status is not None

    assert report.backend_status["is_real_separation"] is False

def test_run_validation_reports_scenario_error():
    class Boom:
        def separate(self, chunk, target, tracks):
            raise RuntimeError("kaboom")

    report = run_validation(Boom(), sample_rate=16000, chunk_samples=400)
    assert not report.passed
    assert all(s.error for s in report.scenarios)
