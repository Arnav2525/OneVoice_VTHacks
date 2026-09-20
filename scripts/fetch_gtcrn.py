

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from urllib.request import urlopen

REVISION = "502ebfab64da7c4a9af78dcb9c6ceef1ebb01c73"
BASE_URL = f"https://raw.githubusercontent.com/Xiaobin-Rong/gtcrn/{REVISION}/"
ASSETS = (
    (
        "stream/onnx_models/gtcrn_simple.onnx",
        "gtcrn_simple.onnx",
        "b4718df6228e7bdf1a8a435cf98f838636eb2fd331acabf86ba87c5192ebcb87",
    ),
    (
        "LICENSE",
        "LICENSE",
        "c467165e5860b4a7494ef4a6e2788e4115d95b5b8989bb5f86287089adddf794",
    ),
)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "checkpoints" / "gtcrn",
    )
    args = parser.parse_args()
    args.dest.mkdir(parents=True, exist_ok=True)
    for remote, name, expected_hash in ASSETS:
        destination = args.dest / name
        if destination.is_file():
            if hashlib.sha256(destination.read_bytes()).hexdigest() == expected_hash:
                print(f"Verified: {destination}")
                continue
            raise RuntimeError(
                f"Existing {destination} has the wrong checksum; refusing to overwrite"
            )
        with urlopen(BASE_URL + remote, timeout=60) as response:
            data = response.read(2 * 1024 * 1024)
        if hashlib.sha256(data).hexdigest() != expected_hash:
            raise RuntimeError(
                f"Checksum verification failed for {name}; no model written"
            )

        with destination.open("xb") as output:
            output.write(data)
        print(f"Downloaded and verified: {destination}")

if __name__ == "__main__":
    main()
