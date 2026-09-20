

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logger = logging.getLogger("onevoice.dolphin.gold_eval")

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Dolphin upstream eval.py runner")
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=Path("data/dolphin_tier_a/tt"),
    )
    parser.add_argument(
        "--conf",
        type=Path,
        default=Path("vendor/dolphin/configs/dolphin.yml"),
    )
    parser.add_argument("--hf-model-id", default="JusperLee/Dolphin")
    parser.add_argument("--segment-seconds", type=float, default=2.0)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)
    repo_root = Path(__file__).resolve().parents[1]
    vendor = repo_root / "vendor" / "dolphin"
    if not (vendor / "eval.py").is_file():
        raise SystemExit(
            "vendor/dolphin missing — run: python scripts/vendor_dolphin.py"
        )
    if not (args.test_dir / "mix.json").is_file():
        raise SystemExit(
            f"test dir missing LRS2 json — run prepare_dolphin_tier_a_data.py "
            f"({args.test_dir})"
        )

    import yaml

    with open(args.conf, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config["use_hf_model"] = True
    config["hf_model_id"] = args.hf_model_id
    config["datamodule"]["data_config"]["test_dir"] = str(
        args.test_dir.resolve()
    )
    config["datamodule"]["data_config"]["train_dir"] = str(args.test_dir.resolve())
    config["datamodule"]["data_config"]["valid_dir"] = str(args.test_dir.resolve())
    config["datamodule"]["data_config"]["segment"] = args.segment_seconds
    config["datamodule"]["data_config"]["batch_size"] = 1
    config["datamodule"]["data_config"]["num_workers"] = 0

    os.chdir(vendor)
    sys.path.insert(0, str(vendor))

    import types

    if "pypesq" not in sys.modules:
        stub = types.ModuleType("pypesq")
        stub.pesq = lambda *args, **kwargs: 0.0  # type: ignore[misc]
        sys.modules["pypesq"] = stub

    from eval import main as eval_main

    logger.info(
        "running Dolphin eval.py on %s (segment=%.1fs)",
        args.test_dir,
        args.segment_seconds,
    )
    eval_main(config)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
