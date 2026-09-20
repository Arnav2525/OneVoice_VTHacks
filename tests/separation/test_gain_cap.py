import numpy as np
import pytest

from onevoice.separation.adapters.dolphin_adapter import DolphinAdapter
from onevoice.separation.config import SeparationConfig

def test_optional_gain_cap_preserves_default_and_limits_boost():
    for cap, expected in [(None, 0.5), (10, 0.02)]:
        adapter = DolphinAdapter(SeparationConfig(params={"max_auto_gain": cap}))
        try:
            output = adapter._apply_auto_gain(np.full(320, 0.002), np.full(320, 0.5))
            np.testing.assert_allclose(output, expected, rtol=1e-6)
            assert not any(adapter._apply_auto_gain(np.zeros(320), np.ones(320)))
        finally:
            adapter.shutdown()

@pytest.mark.parametrize("cap", [0, 0.5, -1, float("nan"), float("inf")])
def test_invalid_gain_cap_rejected(cap):
    with pytest.raises(ValueError, match="max_auto_gain"):
        DolphinAdapter(SeparationConfig(params={"max_auto_gain": cap}))
