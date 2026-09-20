from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from urllib.request import urlopen

BASE_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/"
)
ASSETS = (
    (
        "face_landmarker.task",
        "face_landmarker.task",
        "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff",
    ),
)

MAX_BYTES = 16 * 1024 * 1024


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "checkpoints" / "pause",
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
            data = response.read(MAX_BYTES)
        if hashlib.sha256(data).hexdigest() != expected_hash:
            raise RuntimeError(
                f"Checksum verification failed for {name}; no model written"
            )

        with destination.open("xb") as output:
            output.write(data)
        print(f"Downloaded and verified: {destination}")


if __name__ == "__main__":
    main()
