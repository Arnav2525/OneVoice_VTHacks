from typing import Protocol

from onevoice.core.models.pipeline_result import PipelineResult

class BenchmarkRunner(Protocol):

    def run_benchmark(self) -> None:

        ...

    def collect_result(self, result: PipelineResult) -> None:

        ...

    def generate_report(self) -> str:

        ...
