

from __future__ import annotations

import logging

from bench._common import (
    build_argparser,
    build_pipeline,
    default_targets,
    finalize_run,
    load_config,
    resolve_config_path,
    seed_everything,
    setup_run,
)
from onevoice.telemetry.benchmark_runner import PipelineBenchmarkRunner
from onevoice.telemetry.hardware import collect_hardware_info

def main() -> None:
    args = build_argparser("OneVoice end-to-end latency benchmark").parse_args()
    seed_everything(args.seed)
    config = load_config(resolve_config_path(args.config))
    targets = default_targets(config)

    run = setup_run("latency", config, args.output_dir)
    runner = PipelineBenchmarkRunner(
        name="latency", duration_s=args.duration, targets=targets
    )
    pipeline, chunk_ms = build_pipeline(config, on_result=runner.collect_result,
                                        live=args.live)
    runner.bind(pipeline, chunk_ms)

    runner.run_benchmark()
    aggregate = runner.aggregate()
    hardware = collect_hardware_info()
    report = finalize_run(run, aggregate, targets,
                          per_result_rows=runner.per_result_rows, hardware=hardware)
    print(report)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
