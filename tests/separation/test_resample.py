

import pytest

np = pytest.importorskip("numpy")

from onevoice.separation.utilities import fit_length, resample_waveform

def test_resample_identity_when_rates_match():
    wav = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    out = resample_waveform(wav, 16000, 16000)
    assert list(out) == pytest.approx([0.1, 0.2, 0.3])

def test_downsample_halves_length():
    wav = np.ones(100, dtype=np.float32)
    out = resample_waveform(wav, 16000, 8000)
    assert out.shape[0] == 50

def test_upsample_doubles_length():
    wav = np.ones(50, dtype=np.float32)
    out = resample_waveform(wav, 8000, 16000)
    assert out.shape[0] == 100

def test_resample_empty_is_safe():
    out = resample_waveform(np.array([], dtype=np.float32), 16000, 8000)
    assert out.shape[0] == 0

def test_fit_length_trims():
    out = fit_length(np.arange(10, dtype=np.float32), 4)
    assert list(out) == [0.0, 1.0, 2.0, 3.0]

def test_fit_length_pads():
    out = fit_length(np.array([1.0, 2.0], dtype=np.float32), 5)
    assert list(out) == [1.0, 2.0, 0.0, 0.0, 0.0]

def test_fit_length_noop_when_equal():
    out = fit_length(np.array([1.0, 2.0], dtype=np.float32), 2)
    assert list(out) == [1.0, 2.0]
