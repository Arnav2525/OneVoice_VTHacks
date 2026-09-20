

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

def zip_tree(source: Path, dest: Path, *, prefix: str = "") -> int:

    source = source.resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(source.rglob("*")):
            if not path.is_file():
                continue
            if "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            rel = path.relative_to(source).as_posix()
            arcname = f"{prefix}/{rel}" if prefix else rel
            zf.write(path, arcname)
            count += 1
    return count

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Zip tree for Kaggle upload")
    parser.add_argument("source", type=Path, help="directory to zip")
    parser.add_argument("dest", type=Path, help="output .zip path")
    parser.add_argument(
        "--prefix",
        default="",
        help="optional arcname prefix (e.g. tt for data/dolphin_tier_a/tt)",
    )
    args = parser.parse_args(argv)
    if not args.source.is_dir():
        raise SystemExit(f"not a directory: {args.source}")
    n = zip_tree(args.source, args.dest, prefix=args.prefix.strip("/"))
    print(f"wrote {args.dest} ({n} files)")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
