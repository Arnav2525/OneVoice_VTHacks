

from __future__ import annotations

import array

from demo.session_runtime import _ramp

def _flat(n=320, value=1.0):
    return array.array("f", [value] * n)

def test_fade_out_starts_at_full_and_reaches_silence():
    out = _ramp(_flat(), rising=False, length=128)
    assert out[0] > 0.99
    assert out[127] < 0.01
    assert all(v == 0.0 for v in out[128:])

def test_fade_in_starts_at_silence_and_reaches_full():
    out = _ramp(_flat(), rising=True, length=128)
    assert out[0] < 0.01
    assert out[127] > 0.99
    assert all(v == 1.0 for v in out[128:])

def test_ramp_is_monotonic_in_both_directions():
    rising = _ramp(_flat(), rising=True, length=128)[:128]
    falling = _ramp(_flat(), rising=False, length=128)[:128]
    assert all(b >= a for a, b in zip(rising, rising[1:]))
    assert all(b <= a for a, b in zip(falling, falling[1:]))

def test_no_step_larger_than_the_signal_itself():

    for rising in (True, False):
        out = _ramp(_flat(), rising=rising, length=128)
        steps = [abs(b - a) for a, b in zip(out, out[1:])]
        assert max(steps) < 0.05

def test_ramp_shorter_than_requested_chunk_is_safe():
    out = _ramp(_flat(n=16), rising=False, length=128)
    assert len(out) == 16
    assert out[-1] < out[0]

def test_empty_chunk_is_passed_through():
    assert _ramp(array.array("f", []), rising=True, length=128) == []
