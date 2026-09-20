

from __future__ import annotations

import argparse
import logging
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import soundfile as sf

from bench._dolphin import SAMPLE_RATE
from bench._dolphin_mouth import extract_mouth_npz, probe_frontal_face

logger = logging.getLogger("onevoice.fetch_test_videos")

GITHUB_DEMO = (
    "https://raw.githubusercontent.com/JusperLee/Dolphin/main/real-world-demo"
)

DOLPHIN_DEMO_SINGLES: list[tuple[str, str]] = [
    ("dolphin_demo1_s1", f"{GITHUB_DEMO}/demo1/s1.mp4"),
    ("dolphin_demo1_s2", f"{GITHUB_DEMO}/demo1/s2.mp4"),
    ("dolphin_demo2_s1", f"{GITHUB_DEMO}/demo2/s1.mp4"),
    ("dolphin_demo2_s2", f"{GITHUB_DEMO}/demo2/s2.mp4"),
    ("dolphin_demo3_s1", f"{GITHUB_DEMO}/demo3/s1.mp4"),
    ("dolphin_demo3_s2", f"{GITHUB_DEMO}/demo3/s2.mp4"),
]

DEFAULT_YT_URLS: list[str] = [
    "https://www.youtube.com/watch?v=jNQXAC9IVRw",
    "https://www.youtube.com/watch?v=3AtDnEC4zak",
    "https://www.youtube.com/watch?v=OPf0YbXqDm0",
]

CC0_MIXKIT: list[tuple[str, str]] = [
    ("mixkit_influencer_42323", "https://assets.mixkit.co/videos/42323/42323-720.mp4"),
    ("mixkit_vlogger_41290", "https://assets.mixkit.co/videos/41290/41290-720.mp4"),
    ("mixkit_therapist_4834", "https://assets.mixkit.co/videos/4834/4834-720.mp4"),
    ("mixkit_newscaster_24401", "https://assets.mixkit.co/videos/24401/24401-720.mp4"),
    ("mixkit_businessman_3195", "https://assets.mixkit.co/videos/3195/3195-720.mp4"),
    ("mixkit_presentation_5088", "https://assets.mixkit.co/videos/5088/5088-720.mp4"),
]

def _ffmpeg_exe() -> str:
    path = shutil.which("ffmpeg")
    if path:
        return path
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()

def _trim_video(src: Path, dst: Path, seconds: float) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        _ffmpeg_exe(),
        "-y",
        "-i",
        str(src),
        "-t",
        str(seconds),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        str(dst),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg trim failed: {proc.stderr[-400:]}")

def _has_audio_stream(path: Path) -> bool:
    proc = subprocess.run(
        [_ffmpeg_exe(), "-hide_banner", "-i", str(path)],
        capture_output=True,
        text=True,
    )
    return "Audio:" in (proc.stderr or "")

def _speech_rms(path: Path, *, seconds: float = 2.0) -> float:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav = Path(tmp.name)
    try:
        cmd = [
            _ffmpeg_exe(),
            "-y",
            "-i",
            str(path),
            "-t",
            str(seconds),
            "-ar",
            str(SAMPLE_RATE),
            "-ac",
            "1",
            str(wav),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            return 0.0
        audio, _ = sf.read(str(wav), dtype="float32")
        if len(audio) == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(audio))))
    finally:
        wav.unlink(missing_ok=True)

def _download_url(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 0:
        return
    logger.info("downloading %s", url)
    urllib.request.urlretrieve(url, dest)

def _slug_from_url(url: str) -> str:
    yt = re.search(r"(?:v=|youtu\.be/)([\w-]{6,})", url)
    if yt:
        return f"yt_{yt.group(1)}"
    slug = re.sub(
        r"[^a-zA-Z0-9_-]+", "_", url.split("?")[0].rstrip("/").split("/")[-1]
    )
    return slug[:48] or "yt_clip"

def _fetch_ytdlp(url: str, out_dir: Path, *, section_seconds: float) -> Path | None:
    try:
        import yt_dlp
    except ImportError as exc:
        raise RuntimeError(
            "yt-dlp required for --urls: pip install -e '.[dolphin]'"
        ) from exc

    slug = _slug_from_url(url)
    out_dir.mkdir(parents=True, exist_ok=True)
    end = int(section_seconds)
    opts = {
        "format": "bv*+ba/b",
        "merge_output_format": "mp4",
        "outtmpl": str(out_dir / f"{slug}_raw.%(ext)s"),
        "download_sections": f"*0-{end}",
        "force_keyframes_at_cuts": True,
        "ffmpeg_location": _ffmpeg_exe(),
        "quiet": True,
        "no_warnings": True,
    }
    logger.info("yt-dlp %s (first %ds)", url, end)
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])
    candidates = sorted(out_dir.glob(f"{slug}_raw.*"))
    return candidates[0] if candidates else None

def _validate_mouth_roi(work: Path, *, min_frames: int = 40) -> tuple[bool, str]:
    short = work.with_suffix(".mouth_probe.mp4")
    _trim_video(work, short, 4.0)
    cache = Path(tempfile.mkdtemp(prefix="dolphin_fetch_probe_"))
    try:
        roi = extract_mouth_npz(short, cache_dir=cache)
        short.unlink(missing_ok=True)
        if roi.shape[0] < min_frames:
            return False, f"mouth frames too few ({roi.shape[0]})"
        return True, f"mouth ok ({roi.shape[0]} frames)"
    except Exception as exc:
        short.unlink(missing_ok=True)
        return False, f"mouth pipeline failed: {exc}"
    finally:
        shutil.rmtree(cache, ignore_errors=True)

def _accept_clip(
    src: Path,
    dest: Path,
    *,
    trim_seconds: float,
    force: bool,
    min_speech_rms: float,
    min_speech_rms_dolphin: float,
) -> bool:
    if dest.is_file() and not force:
        if _has_audio_stream(dest) and _speech_rms(dest) >= min_speech_rms:
            ok, reason, _ = probe_frontal_face(dest, min_hit_ratio=0.6)
            if ok:
                logger.info("keep existing %s (%s)", dest.name, reason)
                return True

    work = dest.with_suffix(".work.mp4")
    _trim_video(src, work, trim_seconds)

    if not _has_audio_stream(work):
        logger.warning("SKIP %s: no audio stream", dest.stem)
        work.unlink(missing_ok=True)
        return False

    rms = _speech_rms(work)
    speech_floor = (
        min_speech_rms_dolphin
        if dest.stem.startswith("dolphin_")
        else min_speech_rms
    )
    if rms < speech_floor:
        logger.warning(
            "SKIP %s: speech too quiet in first 2s (rms=%.5f < %.5f)",
            dest.stem,
            rms,
            speech_floor,
        )
        work.unlink(missing_ok=True)
        return False

    ok, reason, hits = probe_frontal_face(work, min_hit_ratio=0.6)
    if not ok:
        if hits > 0:
            ok2, reason2 = _validate_mouth_roi(work)
            if ok2:
                ok, reason = ok2, reason2
            else:
                logger.warning(
                    "SKIP %s: face=%s; mouth=%s", dest.stem, reason, reason2
                )
                work.unlink(missing_ok=True)
                return False
        else:
            logger.warning("SKIP %s: %s", dest.stem, reason)
            work.unlink(missing_ok=True)
            return False

    work.replace(dest)
    logger.info("accepted %s (%s; speech_rms=%.4f)", dest.name, reason, rms)
    return True

def fetch_grid_corpus(
    out_dir: Path,
    *,
    talkers: list[str],
    cache_dir: Path,
    trim_seconds: float,
    force: bool,
    min_speech_rms: float,
) -> list[Path]:

    import os

    from scripts._corpus_grid import build_talker_clip

    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    accepted: list[Path] = []

    for talker in talkers:
        dest = out_dir / f"grid_{talker}.mp4"
        if dest.is_file() and not force:
            ok, reason, _ = probe_frontal_face(dest, min_hit_ratio=0.6)
            rms = _speech_rms(dest)
            if ok and rms >= min_speech_rms:
                logger.info("keep existing %s (%s)", dest.name, reason)
                accepted.append(dest)
                continue

        work = cache_dir / "build" / talker
        logger.info("building GRID clip for %s", talker)
        mp4, _ = build_talker_clip(
            talker,
            cache=cache_dir,
            work=work,
            target_seconds=trim_seconds,
        )
        if _accept_clip(
            mp4,
            dest,
            trim_seconds=trim_seconds,
            force=True,
            min_speech_rms=min_speech_rms,
            min_speech_rms_dolphin=min_speech_rms,
        ):
            accepted.append(dest)

    if len(accepted) < len(talkers):
        raise RuntimeError(
            f"GRID corpus: only {len(accepted)}/{len(talkers)} talkers passed validation"
        )
    return accepted

def fetch_all(
    out_dir: Path,
    *,
    urls: list[str],
    trim_seconds: float,
    min_clips: int,
    force: bool,
    min_speech_rms: float,
    min_speech_rms_dolphin: float,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    accepted: list[Path] = []

    sources: list[tuple[str, str, str]] = []
    if urls:
        sources.extend((f"yt_{_slug_from_url(u)}", u, "ytdlp") for u in urls)
    else:
        for name, url in DOLPHIN_DEMO_SINGLES:
            sources.append((name, url, "direct"))
        for i, url in enumerate(DEFAULT_YT_URLS):
            sources.append((f"yt_default_{i+1}", url, "ytdlp"))
        for name, url in CC0_MIXKIT:
            sources.append((name, url, "direct"))

    for name, source, via in sources:
        if len(accepted) >= min_clips and not force:
            break
        dest = out_dir / f"{name}.mp4"
        if dest.is_file() and not force and dest in accepted:
            continue
        try:
            if via == "direct":
                raw = out_dir / f"_{name}_download.mp4"
                _download_url(source, raw)
                src_path = raw
            else:
                fetched = _fetch_ytdlp(source, out_dir, section_seconds=trim_seconds)
                if fetched is None:
                    logger.error("yt-dlp produced no file for %s", source)
                    continue
                src_path = fetched
            if _accept_clip(
                src_path,
                dest,
                trim_seconds=trim_seconds,
                force=force,
                min_speech_rms=min_speech_rms,
                min_speech_rms_dolphin=min_speech_rms_dolphin,
            ):
                if dest not in accepted:
                    accepted.append(dest)
        except Exception as exc:
            logger.error("FAILED %s: %s", name, exc)

    if len(accepted) < min_clips:
        raise RuntimeError(
            f"only {len(accepted)} clips passed validation (need {min_clips}). "
            "Skipped clips are logged above; pass --urls with frontal solo vlogs."
        )
    return accepted

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fetch single-speaker talking-head clips for Dolphin Tier A",
        epilog=(
            "NOTE: --urls / default YouTube downloads are for LOCAL evaluation only. "
            "Respect site ToS and copyright."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/dolphin_tier_a/raw/single"),
    )
    parser.add_argument(
        "--urls",
        nargs="*",
        default=[],
        help="YouTube/etc URLs (yt-dlp --download-sections first N seconds)",
    )
    parser.add_argument("--trim-seconds", type=float, default=8.0)
    parser.add_argument(
        "--corpus",
        choices=("grid", "cremad", "none"),
        default="none",
        help="research corpus mode (grid=Zenodo GRID AV corpus)",
    )
    parser.add_argument(
        "--talkers",
        nargs="*",
        default=[],
        help="GRID talker ids (default: 8 talkers s1-s4,s5,s6,s22,s23)",
    )
    parser.add_argument(
        "--corpus-cache",
        type=Path,
        default=Path("data/dolphin_tier_a/raw/corpus_cache/grid"),
    )
    parser.add_argument(
        "--min-clips",
        type=int,
        default=8,
        help="minimum accepted clips (8 => 4 mixes / 8 scored targets)",
    )
    parser.add_argument(
        "--min-speech-rms-dolphin",
        type=float,
        default=0.0004,
        help="lower speech floor for Dolphin demo singles (late speech onsets)",
    )
    parser.add_argument(
        "--min-speech-rms",
        type=float,
        default=0.01,
        help="min RMS in first 2s @16kHz (filters silent/music-only clips)",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)

    if args.corpus == "grid":
        from scripts._corpus_grid import DEFAULT_TALKERS

        talkers = args.talkers or list(DEFAULT_TALKERS)
        accepted = fetch_grid_corpus(
            args.out_dir,
            talkers=talkers,
            cache_dir=args.corpus_cache,
            trim_seconds=args.trim_seconds,
            force=args.force,
            min_speech_rms=args.min_speech_rms,
        )
    elif args.corpus == "cremad":
        raise RuntimeError(
            "CREMA-D fallback requires git-lfs clone (~7.5 GB); use --corpus grid "
            "or install CREMA-D locally and point --out-dir manually."
        )
    else:
        accepted = fetch_all(
            args.out_dir,
            urls=args.urls,
            trim_seconds=args.trim_seconds,
            min_clips=args.min_clips,
            force=args.force,
            min_speech_rms=args.min_speech_rms,
            min_speech_rms_dolphin=args.min_speech_rms_dolphin,
        )
    logger.info("accepted %d clips in %s", len(accepted), args.out_dir)
    for p in accepted:
        logger.info("  %s", p.name)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
