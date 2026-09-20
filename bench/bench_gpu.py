

from __future__ import annotations

import logging

from bench._common import (
    build_argparser,
    build_pipeline,
    default_targets,
    finalize_run,
    load_config,
    make_separator,
    resolve_config_path,
    seed_everything,
    setup_run,
)
from onevoice.telemetry.benchmark_runner import PipelineBenchmarkRunner
from onevoice.telemetry.gpu_profiler import NvmlGPUProfiler
from onevoice.telemetry.hardware import collect_hardware_info
from onevoice.telemetry.system_profiler import SystemProfiler, measure_import_time

def main() -> None:
    args = build_argparser("OneVoice GPU/CPU profiling benchmark").parse_args()
    seed_everything(args.seed)
    config = load_config(resolve_config_path(args.config))
    targets = default_targets(config)

    run = setup_run("gpu", config, args.output_dir)

    load_ms = measure_import_time(lambda: make_separator(config))

    gpu = NvmlGPUProfiler()
    sysprof = SystemProfiler()
    runner = PipelineBenchmarkRunner(
        name="gpu",
        duration_s=args.duration,
        targets=targets,
        gpu_profiler=gpu,
        system_profiler=sysprof,
    )
    pipeline, chunk_ms = build_pipeline(config, on_result=runner.collect_result,
                                        live=args.live)
    runner.bind(pipeline, chunk_ms)

    runner.run_benchmark()
    aggregate = runner.aggregate()
    aggregate["resources"]["model_load_ms"] = load_ms
    aggregate["resources"]["gpu_available"] = float(gpu.available)
    hardware = collect_hardware_info()
    report = finalize_run(run, aggregate, targets, hardware=hardware)
    print(report)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
