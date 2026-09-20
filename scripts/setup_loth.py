#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import zipfile
from pathlib import Path

DEFAULT_ROOT = Path.home() / ".cache" / "onevoice" / "LookOnceToHear"
REPO_URL = "https://github.com/vb000/LookOnceToHear.git"
GDRIVE_ID = "1CP0zbZExcqvNLdP9epyhY4fEVp_oQr59"
GDRIVE_URL = f"https://drive.google.com/file/d/{GDRIVE_ID}/view"
REQUIRED_CKPTS = (
    "runs/tsh/best.ckpt",
    "runs/embed/best.ckpt",
)

def _run(cmd: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)

def clone_repo(root: Path) -> None:
    if (root / ".git").exists():
        print(f"repo already cloned at {root}")
        return
    root.parent.mkdir(parents=True, exist_ok=True)
    _run(["git", "clone", "--depth", "1", REPO_URL, str(root)])

def checkpoints_ready(root: Path) -> bool:
    return all((root / rel).is_file() for rel in REQUIRED_CKPTS)

def extract_archive(archive: Path, root: Path) -> None:
    print(f"extracting {archive} -> {root}")
    with zipfile.ZipFile(archive, "r") as zf:
        zf.extractall(root)

def _manual_download_hint() -> None:
    print(
        "\nAutomatic Google Drive download failed (link may be down or quota-limited).\n"
        "Manual steps:\n"
        f"  1. Open: {GDRIVE_URL}\n"
        "  2. Download the zip (if the page says 'file does not exist', the upstream\n"
        "     link is broken — open a GitHub issue at vb000/LookOnceToHear)\n"
        f"  3. Run: python scripts/setup_loth.py --archive <path-to-zip>\n"
        "Expected files after extract:\n"
        "  runs/tsh/best.ckpt\n"
        "  runs/embed/best.ckpt\n",
        file=sys.stderr,
    )

def download_checkpoints(root: Path) -> bool:
    if checkpoints_ready(root):
        print("checkpoints already present")
        return True
    try:
        import gdown
    except ImportError:
        print("gdown not installed; pip install gdown", file=sys.stderr)
        _manual_download_hint()
        return False
    archive = root / "loth_checkpoints.zip"
    print("downloading checkpoints (this may take a few minutes)...")
    urls = [
        f"https://drive.google.com/uc?id={GDRIVE_ID}",
        f"https://drive.google.com/uc?export=download&id={GDRIVE_ID}",
    ]
    last_exc: Exception | None = None
    for url in urls:
        try:
            gdown.download(url=url, output=str(archive), quiet=False, fuzzy=True)
            break
        except Exception as exc:
            last_exc = exc
            print(f"download attempt failed: {exc}")
    else:
        _manual_download_hint()
        if last_exc is not None:
            print(f"last error: {last_exc}", file=sys.stderr)
        return False
    extract_archive(archive, root)
    archive.unlink(missing_ok=True)
    if not checkpoints_ready(root):
        print("archive extracted but expected checkpoints missing", file=sys.stderr)
        _manual_download_hint()
        return False
    for rel in REQUIRED_CKPTS:
        print(f"checkpoint ready: {root / rel}")
    return True

def main() -> int:
    parser = argparse.ArgumentParser(description="Setup LookOnceToHear weights")
    parser.add_argument(
        "--root",
        default=os.environ.get("ONEVOICE_LOTH_ROOT", str(DEFAULT_ROOT)),
        help="clone/download destination",
    )
    parser.add_argument(
        "--archive",
        type=Path,
        help="local zip downloaded manually from Google Drive",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="verify checkpoints exist and exit",
    )
    args = parser.parse_args()
    root = Path(args.root).expanduser().resolve()

    if args.check_only:
        ok = checkpoints_ready(root)
        print("ready" if ok else "missing checkpoints")
        return 0 if ok else 1

    clone_repo(root)
    if args.archive is not None:
        archive = args.archive.expanduser().resolve()
        if not archive.is_file():
            print(f"archive not found: {archive}", file=sys.stderr)
            return 1
        extract_archive(archive, root)
    elif not checkpoints_ready(root):
        if not download_checkpoints(root):
            return 1

    if not checkpoints_ready(root):
        _manual_download_hint()
        return 1

    print("\nSetup complete.")
    print(f"  ONEVOICE_LOTH_ROOT={root}")
    print("  python -m onevoice.streaming --config configs/experiments/look_once_live.yaml --live")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
