from typing import Protocol

class LatencyRecorder(Protocol):

    def record_start(self, metric_name: str, timestamp_ms: float) -> None:

        ...

    def record_end(self, metric_name: str, timestamp_ms: float) -> None:

        ...

    def get_latencies(self) -> dict[str, float]:

        ...
