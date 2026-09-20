

from __future__ import annotations

import array
import threading

class RingBuffer:

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._capacity = capacity
        self._buffer = array.array("f", [0.0] * capacity)
        self._write_pos = 0
        self._read_pos = 0
        self._size = 0
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._not_full = threading.Condition(self._lock)

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def size(self) -> int:
        with self._lock:
            return self._size

    def write(self, samples: array.array, timeout: float | None = None) -> int:

        if samples.typecode != "f":
            raise TypeError("samples must be float32 array.array")

        written = 0
        with self._not_full:
            while written < len(samples):
                if self._size >= self._capacity:
                    if timeout is not None and timeout <= 0:
                        break
                    if not self._not_full.wait(timeout=timeout):
                        break
                if self._size >= self._capacity:
                    break
                self._buffer[self._write_pos] = samples[written]
                self._write_pos = (self._write_pos + 1) % self._capacity
                self._size += 1
                written += 1
            if written:
                self._not_empty.notify_all()
        return written

    def read(self, count: int, timeout: float | None = None) -> array.array:

        out = array.array("f")
        with self._not_empty:
            while len(out) < count:
                if self._size == 0:
                    if timeout is not None and timeout <= 0:
                        break
                    if not self._not_empty.wait(timeout=timeout):
                        break
                if self._size == 0:
                    break
                out.append(self._buffer[self._read_pos])
                self._read_pos = (self._read_pos + 1) % self._capacity
                self._size -= 1
            if out:
                self._not_full.notify_all()
        return out

    def clear(self) -> None:
        with self._lock:
            self._write_pos = 0
            self._read_pos = 0
            self._size = 0
            self._not_empty.notify_all()
            self._not_full.notify_all()
