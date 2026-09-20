

from __future__ import annotations

import argparse
import array
import sys
import wave
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

SAMPLE_RATE = 16_000

def _load_wav_float32(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        if sr != SAMPLE_RATE:
            raise ValueError(
                f"{path} is {sr}Hz, expected {SAMPLE_RATE}Hz -- resample first, "
                f"e.g. ffmpeg -i {path} -ar 16000 -ac 1 {path.stem}_16k.wav"
            )
        n = w.getnframes()
        raw = w.readframes(n)
    return np.asarray(array.array("h", raw), dtype=np.float32) / 32768.0

def _write_wav_float32(path: Path, samples: np.ndarray) -> None:
    clamped = np.clip(samples, -1.0, 1.0)
    pcm16 = (clamped * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm16.tobytes())

def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2) + 1e-12))

def mix_at_snr(target: np.ndarray, interferer: np.ndarray, snr_db: float) -> tuple[np.ndarray, np.ndarray]:

    if len(interferer) < len(target):
        reps = int(np.ceil(len(target) / len(interferer)))
        interferer = np.tile(interferer, reps)
    interferer = interferer[: len(target)]

    target_rms = _rms(target)
    interferer_rms = _rms(interferer)
    if interferer_rms < 1e-8:
        raise ValueError("interferer clip is silent")

    scale = (target_rms / interferer_rms) / (10.0 ** (snr_db / 20.0))
    scaled_interferer = interferer * scale
    mixture = target + scaled_interferer

    peak = float(np.max(np.abs(mixture)))
    if peak > 0.99:
        mixture = mixture / peak * 0.99
    return mixture, scaled_interferer

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-audio", required=True, help="clean target-only WAV (16kHz mono)")
    parser.add_argument("--interferer-audio", required=True, help="any other speaker's WAV (16kHz mono)")
    parser.add_argument("--snr-db", type=float, default=0.0, help="target-vs-interferer SNR (0dB = equal energy, GRID convention)")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)

    target_path = Path(args.target_audio)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    target = _load_wav_float32(target_path)
    interferer = _load_wav_float32(Path(args.interferer_audio))
    mixture, scaled_interferer = mix_at_snr(target, interferer, args.snr_db)

    mixture_path = output_dir / "mixture.wav"
    reference_path = output_dir / "target_reference.wav"
    _write_wav_float32(mixture_path, mixture)
    _write_wav_float32(reference_path, target)

    achieved_snr = 20.0 * np.log10(_rms(target) / max(_rms(scaled_interferer), 1e-8))
    print(f"target:     {target_path}  ({len(target) / SAMPLE_RATE:.1f}s, rms={_rms(target):.4f})")
    print(f"interferer: {args.interferer_audio}  (rms={_rms(interferer):.4f})")
    print(f"requested SNR: {args.snr_db:+.1f}dB | achieved: {achieved_snr:+.1f}dB")
    print(f"wrote mixture   -> {mixture_path}")
    print(f"wrote reference -> {reference_path}  (ground truth for scoring)")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
