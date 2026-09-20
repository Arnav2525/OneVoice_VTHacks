import numpy as np
import pytest

from onevoice.audio.cleanup import RumbleFilter
from onevoice.core.models.audio_chunk import AudioChunk

def chunk(data, epoch=1, channels=1):
    return AudioChunk(
        123.0,
        data,
        16000,
        channels,
        {
            "target_track_id": "a",
            "ui_selection_epoch": epoch,
            "output_ready": True,
        },
    )

def test_reduces_rumble_while_preserving_speech_band():
    t = np.arange(16000) / 16000
    gains = []
    for frequency in (20, 1000):
        source = np.sin(2 * np.pi * frequency * t).astype(np.float32) * 0.1
        result = RumbleFilter().process(chunk(source))
        gains.append(np.std(result.data[1600:]) / np.std(source[1600:]))
        assert result.timestamp_ms == 123.0
        assert result.metadata["output_ready"] is True
    assert gains[0] < 0.3
    assert gains[1] > 0.95

def test_chunk_boundaries_match_continuous_stereo_processing():
    data = np.random.default_rng(3).normal(0, 0.1, 2000).astype(np.float32)
    original = data.copy()
    expected = RumbleFilter().process(chunk(data, channels=2)).data
    filt = RumbleFilter()
    actual = []
    for part in np.split(data, [318, 700, 1400]):
        actual.extend(filt.process(chunk(part, channels=2)).data)
    np.testing.assert_allclose(actual, expected, atol=1e-7)
    np.testing.assert_array_equal(data, original)

def test_new_epoch_or_reset_cannot_leak_previous_voice():
    filt = RumbleFilter()
    filt.process(chunk([0.25] * 320))
    assert not any(filt.process(chunk([0.0] * 320, epoch=2)).data)
    filt.process(chunk([0.25] * 320, epoch=2))
    filt.reset()
    assert not any(filt.process(chunk([0.0] * 320, epoch=2)).data)

@pytest.mark.parametrize("cutoff", [0, -1, float("nan"), float("inf")])
def test_rejects_invalid_cutoff(cutoff):
    with pytest.raises(ValueError):
        RumbleFilter(cutoff)

def test_invalid_audio_and_nyquist_rejected():
    for audio in (chunk([float("nan")]), chunk([0.0], channels=2)):
        with pytest.raises(ValueError):
            RumbleFilter().process(audio)
    with pytest.raises(ValueError):
        RumbleFilter(8000).process(chunk([0.0]))
