

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from bench._dolphin import MOUTH_FPS, ensure_dolphin_import_path, vendor_root

logger = logging.getLogger("onevoice.bench.dolphin.mouth")

MEAN_FACE_REL = Path("assets") / "20words_mean_face.npy"
MEAN_FACE_URL = (
    "https://github.com/JusperLee/Dolphin/raw/main/assets/20words_mean_face.npy"
)

def ensure_mean_face() -> Path:
    root = vendor_root()
    target = root / MEAN_FACE_REL
    if target.is_file():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    import urllib.request

    logger.info("downloading %s", MEAN_FACE_URL)
    urllib.request.urlretrieve(MEAN_FACE_URL, target)
    if not target.is_file():
        raise RuntimeError(f"failed to download mean face landmarks to {target}")
    return target

@contextmanager
def _vendor_cwd():
    root = vendor_root()
    prev = os.getcwd()
    os.chdir(root)
    try:
        yield root
    finally:
        os.chdir(prev)

def extract_mouth_npz(
    video_path: Path,
    *,
    cache_dir: Path | None = None,
    number_of_speakers: int = 1,
    detect_every_n_frame: int = 8,
    face_scale: float = 1.5,
) -> np.ndarray:

    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    ensure_dolphin_import_path()
    ensure_mean_face()
    video_path = video_path.resolve()
    if cache_dir is not None:
        cache_dir = cache_dir.resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"video not found: {video_path}")

    work_root = cache_dir or Path(tempfile.mkdtemp(prefix="dolphin_mouth_"))
    work_root = work_root.resolve()
    work_root.mkdir(parents=True, exist_ok=True)

    with _vendor_cwd():
        from Inference import convert_video_fps, crop_mouth, detectface

        temp_25 = (work_root / "temp_25fps.mp4").resolve()
        convert_video_fps(str(video_path), str(temp_25), target_fps=MOUTH_FPS)

        filename_path = detectface(
            video_input_path=str(temp_25),
            output_path=str(work_root),
            detect_every_N_frame=detect_every_n_frame,
            scalar_face_detection=face_scale,
            number_of_speakers=number_of_speakers,
        )

        mouth_dir = (work_root / "mouthroi").resolve()
        mouth_dir.mkdir(parents=True, exist_ok=True)
        crop_mouth(
            video_direc=str((work_root / "faces").resolve()),
            landmark_direc=str((work_root / "landmark").resolve()),
            filename_path=filename_path,
            save_direc=str(mouth_dir),
            convert_gray=True,
            testset_only=False,
        )

        mouth_npz = mouth_dir / "speaker1.npz"
        if not mouth_npz.is_file():
            raise RuntimeError(
                f"Dolphin mouth crop failed for {video_path}; missing {mouth_npz}"
            )
        data = np.load(mouth_npz)["data"]
        if data.ndim != 3:
            raise RuntimeError(f"unexpected mouth shape {data.shape} for {video_path}")

    if cache_dir is None:
        shutil.rmtree(work_root, ignore_errors=True)

    logger.info(
        "mouth ROI %s: %d frames shape=%s",
        video_path.name,
        data.shape[0],
        data.shape[1:],
    )
    return np.asarray(data, dtype=np.uint8)

def probe_frontal_face(
    video_path: Path,
    *,
    sample_count: int = 5,
    min_hit_ratio: float = 0.8,
) -> tuple[bool, str, int]:

    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    ensure_dolphin_import_path()
    video_path = video_path.resolve()
    if not video_path.is_file():
        return False, f"missing file: {video_path}", 0

    import cv2

    ensure_dolphin_import_path()
    with _vendor_cwd():
        from face_detection_utils import detect_faces

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False, "cannot open video", 0

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total < 10:
        cap.release()
        return False, f"too few frames ({total})", 0

    indices = (
        [0, total // 4, total // 2, (3 * total) // 4, total - 1][:sample_count]
        if total >= sample_count
        else list(range(total))
    )
    hits = 0
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        boxes, _ = detect_faces(
            frame, threshold=0.85, allow_upscaling=True, assume_bgr=True
        )
        if boxes is not None and len(boxes) > 0:
            hits += 1
    cap.release()

    ratio = hits / max(len(indices), 1)
    if ratio < min_hit_ratio:
        return False, f"face detected in {hits}/{len(indices)} sampled frames", hits
    return True, f"face ok ({hits}/{len(indices)} frames)", hits
