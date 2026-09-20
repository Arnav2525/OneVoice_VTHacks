

from __future__ import annotations

import argparse
import array
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from onevoice.telemetry.quality import evaluate_pair

GATE_THRESHOLD_DB = 7.0

def _load_wav_float32(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        n = w.getnframes()
        raw = w.readframes(n)
    return np.asarray(array.array("h", raw), dtype=np.float32) / 32768.0, sr

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, help="target's face on camera (real conditions)")
    parser.add_argument("--mixture", required=True, help="2-speaker mixture WAV (scripts/make_ood_mixture.py output)")
    parser.add_argument("--reference", required=True, help="clean target-only WAV, ground truth for scoring")
    parser.add_argument("--config", default="configs/experiments/dolphin_live.yaml")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "replay_harness.py"),
        "--video", args.video,
        "--audio", args.mixture,
        "--config", args.config,
        "--output-dir", str(output_dir),
        "--log-level", args.log_level,
    ]
    if args.duration is not None:
        cmd += ["--duration", str(args.duration)]

    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout[-8000:])
    if result.returncode != 0:
        print("STDERR:", result.stderr[-4000:])
        print(f"\nreplay_harness.py exited {result.returncode} -- not scoring.")
        return result.returncode

    separated_path = output_dir / "separated_after.wav"
    if not separated_path.is_file():
        print(f"\nmissing {separated_path} -- nothing to score.")
        return 1

    reference, ref_sr = _load_wav_float32(Path(args.reference))
    mixture, mix_sr = _load_wav_float32(Path(args.mixture))
    separated, sep_sr = _load_wav_float32(separated_path)
    assert ref_sr == mix_sr == sep_sr, (ref_sr, mix_sr, sep_sr)

    k = min(len(reference), len(mixture), len(separated))
    metrics = evaluate_pair(reference[:k], separated[:k], mixture[:k], ref_sr)
    snri = metrics["si_snr_improvement_db"]
    passed = snri is not None and snri >= GATE_THRESHOLD_DB

    print("\n" + "=" * 60)
    print("OOD REPLAY CHECK RESULT (held-out material, not GRID corpus)")
    print("=" * 60)
    print(f"config:              {args.config}")
    print(f"SI-SNR improvement:  {snri:+.2f} dB" if snri is not None else "SI-SNR improvement:  N/A")
    print(f"corpus gate (>={GATE_THRESHOLD_DB:.0f}dB): {'PASS' if passed else 'FAIL'}")
    for k_, v in metrics.items():
        if k_ == "si_snr_improvement_db":
            continue
        print(f"{k_}: {v}")
    print(f"\nlisten yourself: {args.mixture}  (input mixture)")
    print(f"listen yourself: {separated_path}  (separated output)")
    print("=" * 60)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
