

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import soundfile as sf

from bench._dolphin import SAMPLE_RATE, infer_separation, load_dolphin, load_test_dataset

logger = logging.getLogger("onevoice.export_dolphin")

def _ffmpeg_exe() -> str:
    path = shutil.which("ffmpeg")
    if path:
        return path
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()

def _mux_mp4(video: Path, wav: Path, out_mp4: Path, seconds: float) -> None:
    cmd = [
        _ffmpeg_exe(),
        "-y",
        "-i",
        str(video),
        "-i",
        str(wav),
        "-t",
        str(seconds),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-shortest",
        str(out_mp4),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg mux failed: {proc.stderr[-400:]}")

def export_target(
    test_dir: Path,
    clip_id: str,
    target: str,
    out_dir: Path,
    *,
    segment_seconds: float,
    video_path: Path | None,
    device: str | None = None,
    fp16: bool = False,
    dolphin_light: bool = False,
) -> Path:
    from bench._dolphin import eval_manifest

    model, dev, _ = load_dolphin(
        warmup=False,
        device=device,
        fp16=fp16,
        dolphin_light=dolphin_light,
    )
    ds = load_test_dataset(test_dir, segment=segment_seconds)
    manifest = eval_manifest(test_dir, segment=segment_seconds)
    want = f"{clip_id}/{target}"

    for idx, meta in enumerate(manifest):
        if meta["clip_id"] != want:
            continue
        mix_t, src_t, mouth_t, _ = ds[idx]
        est = infer_separation(model, mix_t, mouth_t, device=dev, fp16=fp16)
        mix = mix_t.numpy().astype("float32")
        ref = src_t.numpy().astype("float32")
        est = np.asarray(est, dtype=np.float32)
        n = min(len(mix), len(ref), len(est))
        mix, ref, est = mix[:n], ref[:n], est[:n]

        out_dir.mkdir(parents=True, exist_ok=True)
        sf.write(out_dir / "00_mixture_before.wav", mix, SAMPLE_RATE)
        sf.write(out_dir / "01_separated_after.wav", est, SAMPLE_RATE)
        sf.write(out_dir / "02_clean_reference.wav", ref, SAMPLE_RATE)

        if video_path and video_path.is_file():
            sec = n / SAMPLE_RATE
            for stem in (
                "00_mixture_before",
                "01_separated_after",
                "02_clean_reference",
            ):
                _mux_mp4(video_path, out_dir / f"{stem}.wav", out_dir / f"{stem}.mp4", sec)
        logger.info("exported %s -> %s", want, out_dir)
        return out_dir

    raise RuntimeError(f"target not found in manifest: {want}")

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export Dolphin separation samples")
    parser.add_argument("--test-dir", type=Path, default=Path("data/dolphin_tier_a/tt"))
    parser.add_argument("--clip-id", required=True, help="e.g. single_mix_01")
    parser.add_argument("--target", choices=("s1", "s2"), default="s2")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="default: data/dolphin_tier_a/separated_outputs/<clip>_<target>",
    )
    parser.add_argument("--segment-seconds", type=float, default=2.0)
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
    )
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--dolphin-light", action="store_true")
    parser.add_argument(
        "--video",
        type=Path,
        default=None,
        help="optional mp4 for muxed previews (target speaker)",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    out = args.out_dir or Path(
        f"data/dolphin_tier_a/separated_outputs/{args.clip_id}_{args.target}"
    )
    export_target(
        args.test_dir,
        args.clip_id,
        args.target,
        out,
        segment_seconds=args.segment_seconds,
        video_path=args.video,
        device=args.device,
        fp16=args.fp16,
        dolphin_light=args.dolphin_light,
    )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
