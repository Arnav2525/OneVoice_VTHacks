

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

DEFAULT_FILE_ID = "1M4UqK2A2EeHmQ0pCevYqBgaYn3RvklgC"
DEFAULT_DEST = Path(__file__).resolve().parents[1] / "vendor" / "realtse_checkpoints"

VARIANTS = (
    "spk_emb_100",
    "spk_emb_causal_100",
    "tfmap_context_100",
    "tfmap_context_causal_100",
)

def _find_and_place_variants(extract_root: Path, dest: Path) -> list[str]:

    missing: list[str] = []
    for variant in VARIANTS:
        matches = [p for p in extract_root.rglob(variant) if p.is_dir()]
        if not matches:
            missing.append(variant)
            continue
        src = matches[0]
        target = dest / variant
        if target.exists():
            shutil.rmtree(target)
        shutil.move(str(src), str(target))
    return missing

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fetch REAL-TSE pretrained checkpoints from Google Drive"
    )
    parser.add_argument("--file-id", type=str, default=DEFAULT_FILE_ID)
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    args = parser.parse_args(argv)

    try:
        import gdown
    except ImportError as exc:
        raise SystemExit("pip install gdown") from exc

    dest: Path = args.dest
    existing_missing = [
        v for v in VARIANTS if not (dest / v / "avg_model.pt").is_file()
    ]
    if dest.is_dir() and not existing_missing:
        print(f"REAL-TSE checkpoints already present at {dest}")
        return 0

    dest.mkdir(parents=True, exist_ok=True)
    zip_path = dest.parent / "real-tse-ckpt.zip"
    print(f"Downloading Drive file {args.file_id} -> {zip_path}")
    gdown.download(id=args.file_id, output=str(zip_path), quiet=False)

    if not zip_path.is_file() or zip_path.stat().st_size == 0:
        raise SystemExit(f"download failed or produced an empty file at {zip_path}")

    extract_root = dest.parent / "realtse_checkpoints_extracted"
    if extract_root.is_dir():
        shutil.rmtree(extract_root)
    print(f"Extracting {zip_path} -> {extract_root}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_root)

    missing = _find_and_place_variants(extract_root, dest)
    shutil.rmtree(extract_root, ignore_errors=True)
    zip_path.unlink(missing_ok=True)

    if missing:
        print(
            f"WARNING: expected variant folders not found after extraction: {missing}. "
            f"Check the zip layout by hand."
        )
    else:
        print(f"All {len(VARIANTS)} REAL-TSE variants present under {dest}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
