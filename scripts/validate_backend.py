

from __future__ import annotations

import argparse
import logging
import sys
from typing import Any

import yaml

from onevoice.separation.exceptions import BackendError
from onevoice.separation.passthrough import PassthroughSeparator
from onevoice.separation.registry import create_separator
from onevoice.telemetry.audio_validation import run_validation

logger = logging.getLogger("onevoice.validate")

def _load(path: str | None) -> dict[str, Any]:
    if path is None:
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def _audio_params(config: dict[str, Any]) -> tuple[int, int]:
    audio = config.get("audio", {})
    sample_rate = int(audio.get("sample_rate", 16000))
    chunk_samples = int(audio.get("chunk_samples", sample_rate))
    return sample_rate, chunk_samples

def _build_separator(config: dict[str, Any]) -> Any:
    try:
        return create_separator(config.get("backend", {}))
    except BackendError as exc:
        logger.error("backend construction failed (%s); using passthrough", exc)
        return PassthroughSeparator()

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a separation backend.")
    parser.add_argument("--config", default=None, help="experiment YAML path")
    parser.add_argument("--chunk-samples", type=int, default=None, help="override")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config = _load(args.config)
    sample_rate, chunk_samples = _audio_params(config)
    if args.chunk_samples is not None:
        chunk_samples = args.chunk_samples

    separator = _build_separator(config)
    report = run_validation(
        separator,
        sample_rate=sample_rate,
        chunk_samples=chunk_samples,
        seed=args.seed,
    )

    status = report.backend_status or {}
    print("=" * 60)
    print("OneVoice Backend Validation")
    print("=" * 60)
    print(f"backend           : {status.get('backend')}")
    print(f"device            : {status.get('device')}")
    print(f"precision         : {status.get('precision')}")
    print(f"real_separation   : {status.get('is_real_separation')}")
    print(f"load_time_ms      : {status.get('load_time_ms')}")
    print(f"warmup_time_ms    : {status.get('warmup_time_ms')}")
    print(f"avg_infer_ms      : {status.get('avg_infer_ms')}")
    print(f"peak_memory_mb    : {status.get('peak_memory_mb')}")
    print(f"failures          : {status.get('failures')}")
    if status.get("last_error"):
        print(f"last_error        : {status.get('last_error')}")
    if status.get("is_real_separation") is False:
        print("!! WARNING: NOT real separation (passthrough/fallback).")

    print("-" * 60)
    print(f"{'scenario':<20}{'ok':>4}{'len':>5}{'sr':>4}{'clip%':>8}{'peak':>8}")
    print("-" * 60)
    for s in report.scenarios:
        clip_pct = f"{s.clipping_ratio * 100:.2f}"
        print(
            f"{s.name:<20}{('Y' if s.ok else 'N'):>4}"
            f"{('Y' if s.length_ok else 'N'):>5}"
            f"{('Y' if s.sample_rate_ok else 'N'):>4}"
            f"{clip_pct:>8}{s.peak:>8.3f}"
        )
        if s.error:
            print(f"  error: {s.error}")

    print("=" * 60)
    print(f"RESULT: {'PASS' if report.passed else 'FAIL'}")
    print("=" * 60)
    return 0 if report.passed else 1

if __name__ == "__main__":
    sys.exit(main())
