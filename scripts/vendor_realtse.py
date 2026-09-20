

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

REPO_URL = "https://github.com/REAL-TSE/wesep-real-tse.git"
DEFAULT_DEST = Path(__file__).resolve().parents[1] / "vendor" / "realtse"

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Vendor REAL-TSE into vendor/realtse")
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--depth", type=int, default=1, help="shallow clone depth")
    args = parser.parse_args(argv)

    dest: Path = args.dest
    if (dest / "wesep").is_dir():
        print(f"REAL-TSE already present at {dest}")
        return 0

    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["git", "clone", "--depth", str(args.depth), REPO_URL, str(dest)]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"Cloned REAL-TSE to {dest}")
    print(
        "NOTE: install deps separately: "
        "pip install torch torchaudio && pip install -r requirements.txt && "
        "pip install git+https://github.com/wenet-e2e/wespeaker.git@8f53b6485d9f88a207bd17e7f8dba899495ec794"
    )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
