

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from bench._dolphin_mouth import probe_frontal_face

logger = logging.getLogger("onevoice.prune_dolphin")

ALWAYS_HARD: tuple[str, ...] = (
    "dolphin_demo1_s1.mp4",
    "yt_default_1.mp4",
)

def prune(
    single_dir: Path,
    hard_dir: Path,
    *,
    probe_demo2: bool = True,
    corpus_only: bool = False,
) -> list[str]:
    hard_dir.mkdir(parents=True, exist_ok=True)
    moved: list[str] = []

    for name in ALWAYS_HARD:
        src = single_dir / name
        if src.is_file():
            dst = hard_dir / name
            shutil.move(str(src), str(dst))
            moved.append(name)
            logger.info("hard_case (blocked): %s", name)

    if probe_demo2:
        for name in ("dolphin_demo2_s1.mp4", "dolphin_demo2_s2.mp4"):
            src = single_dir / name
            if not src.is_file():
                continue
            ok, reason, _ = probe_frontal_face(src, min_hit_ratio=0.6)
            if ok:
                logger.info("keep in pool: %s (%s)", name, reason)
                continue
            dst = hard_dir / name
            shutil.move(str(src), str(dst))
            moved.append(name)
            logger.warning("hard_case (probe fail): %s — %s", name, reason)

    if corpus_only:
        for src in sorted(single_dir.glob("*.mp4")):
            if src.name.startswith("grid_"):
                continue
            if src.name.startswith("_"):
                continue
            dst = hard_dir / src.name
            shutil.move(str(src), str(dst))
            moved.append(src.name)
            logger.info("hard_case (non-corpus): %s", src.name)

    return moved

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prune Dolphin scored clip pool")
    parser.add_argument(
        "--single-dir",
        type=Path,
        default=Path("data/dolphin_tier_a/raw/single"),
    )
    parser.add_argument(
        "--hard-dir",
        type=Path,
        default=Path("data/dolphin_tier_a/hard_cases"),
    )
    parser.add_argument("--skip-demo2-probe", action="store_true")
    parser.add_argument(
        "--corpus-only",
        action="store_true",
        help="move all non-grid_* clips to hard_cases",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    moved = prune(
        args.single_dir,
        args.hard_dir,
        probe_demo2=not args.skip_demo2_probe,
        corpus_only=args.corpus_only,
    )
    logger.info("moved %d clip(s) to %s", len(moved), args.hard_dir)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
