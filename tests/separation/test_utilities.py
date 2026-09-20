import array
import time

import pytest

from onevoice.core.models.audio_chunk import AudioChunk
from onevoice.separation.exceptions import InferenceTimeoutError
from onevoice.separation.utilities import (
    chunk_to_float_list,
    make_output_chunk,
    run_with_timeout,
    select_device,
    silent_chunk,
)

def test_chunk_to_float_list_from_array():
    assert chunk_to_float_list(array.array("f", [1.0, 2.0])) == [1.0, 2.0]

def test_chunk_to_float_list_from_list():
    assert chunk_to_float_list([1, 2, 3]) == [1.0, 2.0, 3.0]

def test_chunk_to_float_list_none():
    assert chunk_to_float_list(None) == []

def test_silent_chunk_shape():
    chunk = silent_chunk(16000, 320)
    assert len(chunk.data) == 320
    assert chunk.sample_rate == 16000

def test_make_output_chunk_preserves_stream():
    ref = AudioChunk(
        timestamp_ms=5.0, data=[0.0], sample_rate=8000, channels=2, metadata={"a": 1}
    )
    out = make_output_chunk([0.1, 0.2], ref, {"b": 2})
    assert out.sample_rate == 8000
    assert out.channels == 2
    assert out.metadata == {"a": 1, "b": 2}
    assert out.timestamp_ms == 5.0

def test_select_device_cpu():
    assert select_device("cpu") == "cpu"

def test_select_device_auto_returns_valid():
    assert select_device("auto") in {"cpu", "cuda:0"}

def test_run_with_timeout_returns_value():
    assert run_with_timeout(lambda: 42, timeout_s=1.0) == 42

def test_run_with_timeout_raises_on_overrun():
    def slow():
        time.sleep(0.5)
        return 1

    with pytest.raises(InferenceTimeoutError):
        run_with_timeout(slow, timeout_s=0.05)
