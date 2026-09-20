

import array
import threading

import pytest

from onevoice.audio.ring_buffer import RingBuffer

def _arr(values):
    return array.array("f", values)

def test_write_then_read_roundtrip():
    rb = RingBuffer(8)
    assert rb.write(_arr([1.0, 2.0, 3.0])) == 3
    assert rb.size == 3
    out = rb.read(3, timeout=0.1)
    assert list(out) == [1.0, 2.0, 3.0]
    assert rb.size == 0

def test_write_beyond_capacity_is_partial():
    rb = RingBuffer(4)
    written = rb.write(_arr([1, 2, 3, 4, 5, 6]), timeout=0)
    assert written == 4
    assert rb.size == 4

def test_read_more_than_available_non_blocking():
    rb = RingBuffer(8)
    rb.write(_arr([1, 2]))
    out = rb.read(5, timeout=0)
    assert list(out) == [1.0, 2.0]

def test_read_empty_non_blocking_returns_empty():
    rb = RingBuffer(4)
    assert list(rb.read(3, timeout=0)) == []

def test_wraparound_preserves_order():
    rb = RingBuffer(4)
    rb.write(_arr([1, 2, 3]))
    assert list(rb.read(2, timeout=0)) == [1.0, 2.0]
    rb.write(_arr([4, 5, 6]))
    assert list(rb.read(4, timeout=0)) == [3.0, 4.0, 5.0, 6.0]

def test_clear_resets():
    rb = RingBuffer(4)
    rb.write(_arr([1, 2, 3]))
    rb.clear()
    assert rb.size == 0
    assert list(rb.read(3, timeout=0)) == []

def test_wrong_typecode_rejected():
    rb = RingBuffer(4)
    with pytest.raises(TypeError):
        rb.write(array.array("i", [1, 2, 3]))

def test_zero_capacity_rejected():
    with pytest.raises(ValueError):
        RingBuffer(0)

def test_threaded_producer_consumer():
    rb = RingBuffer(16)
    total = 2000
    consumed = []

    def producer():
        remaining = total
        while remaining > 0:
            n = rb.write(_arr([1.0] * min(8, remaining)), timeout=1.0)
            remaining -= n

    def consumer():
        while len(consumed) < total:
            out = rb.read(8, timeout=1.0)
            consumed.extend(out)

    t1 = threading.Thread(target=producer)
    t2 = threading.Thread(target=consumer)
    t1.start()
    t2.start()
    t1.join(timeout=5)
    t2.join(timeout=5)
    assert len(consumed) == total
    assert all(v == 1.0 for v in consumed)
