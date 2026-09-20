

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import urllib.request
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import soundfile as sf

from bench._dolphin import SAMPLE_RATE
from bench._dolphin_mouth import extract_mouth_npz

logger = logging.getLogger("onevoice.prepare_dolphin")

GITHUB_RAW = (
    "https://raw.githubusercontent.com/JusperLee/Dolphin/main/real-world-demo"
)
DEMOS = ("demo1", "demo2", "demo3")
MOUTH_FPS = 25

def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 0:
        logger.info("skip existing %s", dest.name)
        return
    logger.info("downloading %s", url)
    urllib.request.urlretrieve(url, dest)

def _ffmpeg_exe() -> str:
    import shutil

    path = shutil.which("ffmpeg")
    if path:
        return path
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as exc:
        raise RuntimeError(
            "ffmpeg required for mp4 audio extraction: pip install imageio-ffmpeg"
        ) from exc

def _extract_audio_wav(video_path: Path, wav_path: Path) -> np.ndarray:
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        _ffmpeg_exe(),
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(SAMPLE_RATE),
        "-ac",
        "1",
        str(wav_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed on {video_path}: {proc.stderr[-500:]}"
        )
    audio, _ = sf.read(str(wav_path), dtype="float32")
    return np.asarray(audio, dtype=np.float32)

def _trim_mouth(mouth: np.ndarray, n_audio_samples: int) -> np.ndarray:
    fps_len = int(round(n_audio_samples / SAMPLE_RATE * MOUTH_FPS))
    if fps_len <= 0:
        raise RuntimeError("audio too short for mouth alignment")
    if mouth.shape[0] < fps_len:
        pad = np.repeat(mouth[-1:], fps_len - mouth.shape[0], axis=0)
        logger.warning(
            "padding mouth %d -> %d frames for A/V drift",
            mouth.shape[0],
            fps_len,
        )
        mouth = np.concatenate([mouth, pad], axis=0)
    return mouth[:fps_len]

def _trim_video(src: Path, dst: Path, max_seconds: float) -> Path:

    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        _ffmpeg_exe(),
        "-y",
        "-i",
        str(src),
        "-t",
        str(max_seconds),
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(dst),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg trim failed on {src}: {proc.stderr[-500:]}")
    return dst

def prepare_controlled_pair(
    s1_vid: Path,
    s2_vid: Path,
    clip_id: str,
    out_dir: Path,
    *,
    mouth_cache: Path,
    max_seconds: float | None,
    force: bool,
) -> tuple[str, int]:

    clip_dir = out_dir / clip_id
    clip_dir.mkdir(parents=True, exist_ok=True)

    s1_wav = clip_dir / "s1.wav"
    s2_wav = clip_dir / "s2.wav"
    mix_wav = clip_dir / "mix.wav"
    s1_mouth = clip_dir / "s1_mouth.npz"
    s2_mouth = clip_dir / "s2_mouth.npz"

    s1_audio = _extract_audio_wav(s1_vid, s1_wav)
    s2_audio = _extract_audio_wav(s2_vid, s2_wav)
    n = min(len(s1_audio), len(s2_audio))
    if max_seconds is not None:
        cap = int(max_seconds * SAMPLE_RATE)
        n = min(n, cap)
    s1 = s1_audio[:n]
    s2 = s2_audio[:n]
    mix = s1 + s2

    sf.write(str(s1_wav), s1, SAMPLE_RATE)
    sf.write(str(s2_wav), s2, SAMPLE_RATE)
    sf.write(str(mix_wav), mix, SAMPLE_RATE)

    cache = mouth_cache / clip_id
    trim_dir = cache / "_trim"
    v1, v2 = s1_vid, s2_vid
    if max_seconds is not None:
        v1 = _trim_video(s1_vid, trim_dir / "s1_trim.mp4", max_seconds)
        v2 = _trim_video(s2_vid, trim_dir / "s2_trim.mp4", max_seconds)

    if force or not s1_mouth.is_file():
        s1_roi = _trim_mouth(extract_mouth_npz(v1, cache_dir=cache / "s1"), n)
        np.savez_compressed(s1_mouth, data=s1_roi)
    if force or not s2_mouth.is_file():
        s2_roi = _trim_mouth(extract_mouth_npz(v2, cache_dir=cache / "s2"), n)
        np.savez_compressed(s2_mouth, data=s2_roi)

    return clip_id, n

def prepare_controlled_clip(
    demo: str,
    raw_dir: Path,
    out_dir: Path,
    *,
    mouth_cache: Path,
    max_seconds: float | None,
    force: bool,
) -> tuple[str, int]:

    clip_id = demo
    clip_dir = out_dir / clip_id
    clip_dir.mkdir(parents=True, exist_ok=True)

    for name in ("s1", "s2"):
        src = raw_dir / demo / f"{name}.mp4"
        if not src.is_file():
            url = f"{GITHUB_RAW}/{demo}/{name}.mp4"
            _download(url, src)

    return prepare_controlled_pair(
        raw_dir / demo / "s1.mp4",
        raw_dir / demo / "s2.mp4",
        clip_id,
        out_dir,
        mouth_cache=mouth_cache,
        max_seconds=max_seconds,
        force=force,
    )

PREFERRED_SINGLE_ORDER: tuple[str, ...] = (
    "dolphin_demo1_s2.mp4",
    "dolphin_demo3_s2.mp4",
    "dolphin_demo2_s2.mp4",
    "dolphin_demo2_s1.mp4",
    "dolphin_demo1_s1.mp4",
    "yt_default_1.mp4",
)

def _order_singles(singles: list[Path]) -> list[Path]:
    by_name = {p.name: p for p in singles}
    ordered = [by_name[n] for n in PREFERRED_SINGLE_ORDER if n in by_name]
    ordered.extend(p for p in singles if p not in ordered)
    return ordered

def prepare_from_singles(
    single_dir: Path,
    out_dir: Path,
    *,
    mouth_cache: Path,
    max_seconds: float | None,
    force: bool,
    mix_prefix: str = "single_mix",
) -> list[tuple[str, int]]:

    singles = sorted(
        p
        for p in single_dir.glob("*.mp4")
        if not p.name.startswith("_") and "_raw" not in p.stem
    )
    if len(singles) < 2:
        raise RuntimeError(
            f"need at least 2 single-speaker clips in {single_dir}; "
            "run: python scripts/fetch_test_videos.py --corpus grid"
        )

    pairs: list[tuple[Path, Path]] = []
    grid_names = {p.stem for p in singles if p.name.startswith("grid_")}
    if len(grid_names) >= 8:
        from scripts._corpus_grid import CORPUS_PAIR_ORDER

        by_stem = {p.stem: p for p in singles}
        for a, b in CORPUS_PAIR_ORDER:
            if a not in by_stem or b not in by_stem:
                raise RuntimeError(f"corpus pair missing: {a} + {b}")
            pairs.append((by_stem[a], by_stem[b]))
    else:
        singles = _order_singles(singles)
        for i in range(0, len(singles) - 1, 2):
            pairs.append((singles[i], singles[i + 1]))
        if len(singles) % 2:
            logger.warning(
                "odd number of singles (%d); pairing last with first for extra mix",
                len(singles),
            )
            pairs.append((singles[-1], singles[0]))

    clips: list[tuple[str, int]] = []
    for i, (v1, v2) in enumerate(pairs):
        clip_id = f"{mix_prefix}_{i + 1:02d}"
        clip_id, n = prepare_controlled_pair(
            v1,
            v2,
            clip_id,
            out_dir,
            mouth_cache=mouth_cache,
            max_seconds=max_seconds,
            force=force,
        )
        clips.append((clip_id, n))
        logger.info(
            "paired %s + %s -> %s (%d samples)",
            v1.name,
            v2.name,
            clip_id,
            n,
        )
    return clips

def write_json_index(out_dir: Path, clips: list[tuple[str, int]]) -> None:
    mix_infos: list[list[object]] = []
    s1_infos: list[list[object]] = []
    s2_infos: list[list[object]] = []

    for clip_id, n_samples in clips:
        base = (out_dir / clip_id).resolve()
        mix_infos.append([str((base / "mix.wav").as_posix()), n_samples])
        s1_infos.append(
            [
                str((base / "s1.wav").as_posix()),
                str((base / "s1_mouth.npz").as_posix()),
                n_samples,
            ]
        )
        s2_infos.append(
            [
                str((base / "s2.wav").as_posix()),
                str((base / "s2_mouth.npz").as_posix()),
                n_samples,
            ]
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in (
        ("mix", mix_infos),
        ("s1", s1_infos),
        ("s2", s2_infos),
    ):
        with open(out_dir / f"{name}.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prepare controlled Dolphin Tier A LRS2 test clips"
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/dolphin_tier_a/tt"),
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/dolphin_tier_a/raw"),
    )
    parser.add_argument(
        "--mouth-cache",
        type=Path,
        default=Path("data/dolphin_tier_a/mouth_cache"),
    )
    parser.add_argument(
        "--demos",
        nargs="*",
        default=list(DEMOS),
        help="real-world-demo folder names (default: demo1 demo2 demo3)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="rebuild mouth npz even if clip files exist",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=8.0,
        help="trim clips to this duration before mouth extraction (None=full)",
    )
    parser.add_argument(
        "--from-singles",
        action="store_true",
        help="build mixes from data/dolphin_tier_a/raw/single/*.mp4 pairs",
    )
    parser.add_argument(
        "--single-dir",
        type=Path,
        default=Path("data/dolphin_tier_a/raw/single"),
    )
    parser.add_argument(
        "--mix-prefix",
        type=str,
        default="corpus_mix",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)

    clips: list[tuple[str, int]] = []
    if args.from_singles:
        clips = prepare_from_singles(
            args.single_dir,
            args.out_dir,
            mouth_cache=args.mouth_cache,
            max_seconds=args.max_seconds,
            force=args.force,
            mix_prefix=args.mix_prefix,
        )
    else:
        for demo in args.demos:
            clip_id, n = prepare_controlled_clip(
                demo,
                args.raw_dir,
                args.out_dir,
                mouth_cache=args.mouth_cache,
                max_seconds=args.max_seconds,
                force=args.force,
            )
            clips.append((clip_id, n))
            logger.info("prepared controlled clip %s (%d samples)", clip_id, n)

    write_json_index(args.out_dir, clips)
    logger.info("wrote LRS2 index to %s", args.out_dir)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
