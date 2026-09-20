from typing import Protocol

class GPUProfiler(Protocol):

    def start_profiling(self) -> None:

        ...

    def stop_profiling(self) -> None:

        ...

    def get_metrics(self) -> dict[str, float]:

        ...
