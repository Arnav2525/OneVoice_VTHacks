

from __future__ import annotations

import logging
import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf

from bench._dolphin import SAMPLE_RATE

logger = logging.getLogger("onevoice.corpus.grid")

ZENODO_RECORD = "3625687"
ZENODO_FILE = f"https://zenodo.org/api/records/{ZENODO_RECORD}/files"

MALE_TALKERS: tuple[str, ...] = (
    "s1",
    "s2",
    "s3",
    "s4",
    "s7",
    "s8",
    "s9",
    "s10",
    "s11",
    "s12",
    "s13",
    "s14",
    "s15",
    "s16",
    "s17",
    "s18",
    "s19",
    "s20",
)
FEMALE_TALKERS: tuple[str, ...] = (
    "s5",
    "s6",
    "s22",
    "s23",
    "s24",
    "s25",
    "s26",
    "s27",
    "s28",
    "s29",
    "s30",
    "s31",
    "s32",
    "s33",
    "s34",
)

DEFAULT_TALKERS: tuple[str, ...] = (
    "s1", "s2", "s3", "s4", "s7", "s8", "s9", "s10", "s11", "s12", "s13",
    "s14", "s15", "s16", "s17", "s18", "s19",
    "s5", "s6", "s22", "s23", "s24", "s25", "s26", "s27", "s28", "s29",
    "s30", "s31", "s32", "s33", "s34",
)

CORPUS_PAIR_ORDER: tuple[tuple[str, str], ...] = (
    ("grid_s1", "grid_s5"),
    ("grid_s2", "grid_s6"),
    ("grid_s3", "grid_s22"),
    ("grid_s4", "grid_s23"),
    ("grid_s7", "grid_s24"),
    ("grid_s8", "grid_s25"),
    ("grid_s9", "grid_s26"),
    ("grid_s10", "grid_s27"),
    ("grid_s11", "grid_s28"),
    ("grid_s12", "grid_s29"),
    ("grid_s13", "grid_s30"),
    ("grid_s14", "grid_s31"),
    ("grid_s15", "grid_s32"),
    ("grid_s16", "grid_s33"),
    ("grid_s17", "grid_s34"),
    ("grid_s18", "grid_s19"),
)

def _ffmpeg_exe() -> str:
    path = shutil.which("ffmpeg")
    if path:
        return path
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()

def _run_ffmpeg(cmd: list[str], *, label: str) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed ({label}): {proc.stderr[-500:]}")

def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 1_000_000:
        logger.info("skip existing %s", dest.name)
        return
    logger.info("downloading %s", dest.name)

    def _report(block_num: int, block_size: int, total_size: int) -> None:
        if total_size > 0 and block_num % 5000 == 0:
            pct = min(100, block_num * block_size * 100 // total_size)
            logger.info("  %s: ~%d%%", dest.name, pct)

    urllib.request.urlretrieve(url, dest, reporthook=_report)

def _ensure_video_zip(talker: str, cache: Path) -> Path:
    dest = cache / f"{talker}.zip"
    _download(f"{ZENODO_FILE}/{talker}.zip/content", dest)
    return dest

def _ensure_audio_zip(cache: Path) -> Path:
    dest = cache / "audio_25k.zip"
    _download(f"{ZENODO_FILE}/audio_25k.zip/content", dest)
    return dest

def _extract_video(talker: str, cache: Path) -> Path:
    zpath = _ensure_video_zip(talker, cache)
    out = cache / "video" / talker
    if out.is_dir() and any(out.rglob("*.mpg")):
        return out
    if out.is_dir():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zpath) as zf:
        for name in zf.namelist():
            if name.endswith("/") or "__MACOSX" in name:
                continue
            if not name.endswith(".mpg"):
                continue
            target = out / Path(name.replace("\\", "/"))
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.is_file():
                continue
            with zf.open(name) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
    return out

def _extract_talkers_audio(talkers: list[str], cache: Path) -> Path:
    audio_root = cache / "audio"
    needed = [audio_root / t for t in talkers]
    if all(d.is_dir() and any(d.glob("*.wav")) for d in needed):
        return audio_root
    zpath = _ensure_audio_zip(cache)
    audio_root.mkdir(parents=True, exist_ok=True)
    talker_set = set(talkers)
    with zipfile.ZipFile(zpath) as zf:
        for name in zf.namelist():
            norm = name.replace("\\", "/")
            if norm.endswith("/") or "__MACOSX" in norm or not norm.endswith(".wav"):
                continue
            parts = norm.split("/")

            if len(parts) < 3 or parts[0] != "audio_25k":
                continue
            talker, fname = parts[1], parts[2]
            if talker not in talker_set:
                continue
            target = audio_root / talker / fname
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.is_file():
                continue
            with zf.open(name) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
    return audio_root

def _list_utterances(video_dir: Path, talker: str) -> list[str]:
    utts: set[str] = set()
    for mpg in video_dir.rglob("*.mpg"):
        if mpg.name.startswith("._"):
            continue
        utts.add(mpg.stem)
    return sorted(utts)

def _utterance_mpg(video_dir: Path, utt: str) -> Path | None:
    hits = sorted(p for p in video_dir.rglob(f"{utt}.mpg") if not p.name.startswith("._"))
    return hits[0] if hits else None

def _utterance_wav(audio_root: Path, talker: str, utt: str) -> Path | None:
    cand = audio_root / talker / f"{utt}.wav"
    if cand.is_file():
        return cand
    hits = sorted(audio_root.rglob(f"{talker}/{utt}.wav"))
    return hits[0] if hits else None

def _resample_mono_16k(wav_in: Path, wav_out: Path) -> None:
    audio, sr = sf.read(str(wav_in), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        n_out = int(round(len(audio) * SAMPLE_RATE / sr))
        x_old = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=max(n_out, 1), endpoint=False)
        audio = np.interp(x_new, x_old, audio).astype(np.float32)
    sf.write(str(wav_out), audio, SAMPLE_RATE)

def _mux_utterance(mpg: Path, seg_out: Path) -> None:

    cmd = [
        _ffmpeg_exe(),
        "-y",
        "-i",
        str(mpg),
        "-map",
        "0:v:0",
        "-map",
        "0:a:0",
        "-r",
        "25",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-ar",
        str(SAMPLE_RATE),
        "-ac",
        "1",
        str(seg_out),
    ]
    _run_ffmpeg(cmd, label=f"mux {mpg.name}")

def _concat_segments(segments: list[Path], out_mp4: Path) -> None:
    list_file = out_mp4.parent / "concat.txt"
    with open(list_file, "w", encoding="utf-8") as fh:
        for seg in segments:
            fh.write(f"file '{seg.resolve().as_posix()}'\n")
    cmd = [
        _ffmpeg_exe(),
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_file),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        str(out_mp4),
    ]
    _run_ffmpeg(cmd, label="concat segments")

def build_talker_clip(
    talker: str,
    *,
    cache: Path,
    work: Path,
    utterances_per_clip: int = 3,
    target_seconds: float = 8.0,
) -> tuple[Path, Path]:

    video_dir = _extract_video(talker, cache)
    utts = _list_utterances(video_dir, talker)
    if len(utts) < utterances_per_clip:
        raise RuntimeError(f"{talker}: need {utterances_per_clip} utts, found {len(utts)}")

    work.mkdir(parents=True, exist_ok=True)
    segments_dir = work / "segments"
    if segments_dir.is_dir():
        shutil.rmtree(segments_dir)
    segments_dir.mkdir()

    segments: list[Path] = []
    for i, utt in enumerate(utts[:utterances_per_clip]):
        mpg = _utterance_mpg(video_dir, utt)
        if mpg is None:
            raise RuntimeError(f"missing mpg for {talker}/{utt}")
        seg = segments_dir / f"seg_{i:02d}.mp4"
        _mux_utterance(mpg, seg)
        segments.append(seg)

    raw_mp4 = work / f"grid_{talker}_raw.mp4"
    _concat_segments(segments, raw_mp4)

    mp4_out = work / f"grid_{talker}.mp4"
    wav_out = work / "audio.wav"
    cmd = [
        _ffmpeg_exe(),
        "-y",
        "-i",
        str(raw_mp4),
        "-t",
        str(target_seconds),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-ar",
        str(SAMPLE_RATE),
        "-ac",
        "1",
        str(mp4_out),
    ]
    _run_ffmpeg(cmd, label=f"trim {talker}")

    cmd_wav = [
        _ffmpeg_exe(),
        "-y",
        "-i",
        str(mp4_out),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(SAMPLE_RATE),
        "-ac",
        "1",
        str(wav_out),
    ]
    _run_ffmpeg(cmd_wav, label=f"extract wav {talker}")
    return mp4_out, wav_out
