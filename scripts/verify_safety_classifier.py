

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from demo.safety import DEFAULT_MONITORED_CLASSES, YamnetClassifier  # noqa: E402

def _load_audio_mono16k(path: Path) -> np.ndarray:

    import imageio_ffmpeg

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    proc = subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-i",
            str(path),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-f",
            "s16le",
            "-",
        ],
        capture_output=True,
        check=True,
    )
    return np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0

def _windows(samples: np.ndarray, sample_rate: int, window_s: float, hop_s: float):
    window_n = int(window_s * sample_rate)
    hop_n = int(hop_s * sample_rate)
    start = 0
    while start + window_n <= len(samples):
        yield start / sample_rate, samples[start : start + window_n]
        start += hop_n

def run(path: Path, activate_thresh: float, window_s: float, hop_s: float) -> dict:
    samples = _load_audio_mono16k(path)
    duration_s = len(samples) / 16_000
    print(f"{path} -- {duration_s:.1f}s @ 16kHz")

    diagnostic = YamnetClassifier(monitored_classes=None)
    monitored = YamnetClassifier(monitored_classes=set(DEFAULT_MONITORED_CLASSES))

    best_monitored = 0.0
    best_monitored_name = ""
    per_class_max = dict.fromkeys(DEFAULT_MONITORED_CLASSES, 0.0)

    n_windows = 0
    for t_s, window in _windows(samples, 16_000, window_s, hop_s):
        n_windows += 1
        top = diagnostic.classify(window, 16_000)
        event = monitored.classify(window, 16_000)
        if event is not None and event.confidence > best_monitored:
            best_monitored = event.confidence
            best_monitored_name = event.class_name
        if event is not None:
            per_class_max[event.class_name] = max(
                per_class_max[event.class_name], event.confidence
            )
        crosses = event is not None and event.confidence >= activate_thresh
        flag = " <-- CROSSES THRESHOLD" if crosses else ""
        best_str = f"{event.class_name!r} ({event.confidence:.2f})" if event else "-"
        print(
            f"  t={t_s:5.2f}s  top-1={top.class_name!r} ({top.confidence:.2f})"
            f"  best-monitored={best_str}{flag}"
        )

    print()
    print(f"{n_windows} windows checked.")
    print(
        f"Best monitored-class confidence anywhere: "
        f"{best_monitored_name!r} ({best_monitored:.2f})"
    )
    for name, conf in per_class_max.items():
        print(f"  max({name!r}) = {conf:.2f}")
    crossed = best_monitored >= activate_thresh
    print(f"Would activate override (thresh={activate_thresh}): {crossed}")
    return {
        "crossed": crossed,
        "best_confidence": best_monitored,
        "best_class": best_monitored_name,
    }

def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--negative", type=Path, help="a clip that must NOT trigger (e.g. plain speech)"
    )
    group.add_argument(
        "--positive",
        type=Path,
        help="a clip that SHOULD trigger (siren/alarm/baby cry)",
    )
    parser.add_argument("--activate-thresh", type=float, default=0.5)
    parser.add_argument("--window-s", type=float, default=0.96)
    parser.add_argument("--hop-s", type=float, default=0.48)
    return parser

def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    path = args.negative or args.positive
    result = run(path, args.activate_thresh, args.window_s, args.hop_s)

    print()
    if args.negative:
        ok = not result["crossed"]
        verdict = (
            "PASS -- never crossed threshold" if ok else "FAIL -- spuriously triggered"
        )
    else:
        ok = result["crossed"]
        verdict = "PASS -- triggered as expected" if ok else "FAIL -- never triggered"
    print(verdict)
    return 0 if ok else 1

if __name__ == "__main__":
    raise SystemExit(main())
