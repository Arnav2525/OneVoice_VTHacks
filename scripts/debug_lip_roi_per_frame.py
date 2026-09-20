

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import cv2
import numpy as np

from onevoice.core.models.frame import Frame
from onevoice.video.detectors.mediapipe_detector import MediaPipeFaceDetector
from onevoice.video.trackers.iou_tracker import IouFaceTracker, TrackerConfig

MOUTH_FPS = 25
WINDOW_FRAMES = int(1.5 * MOUTH_FPS)

def main() -> None:
    clip_id, target, out_dir, start_frame = (
        sys.argv[1],
        sys.argv[2],
        Path(sys.argv[3]),
        int(sys.argv[4]),
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    video_path = REPO_ROOT / "data" / "dolphin_tier_a" / "mouth_cache" / clip_id / target / "temp_25fps.mp4"
    cap = cv2.VideoCapture(str(video_path))
    frames_bgr = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames_bgr.append(frame)
    cap.release()

    detector = MediaPipeFaceDetector()
    tracker = IouFaceTracker(detector, TrackerConfig(min_hits=1))

    window = frames_bgr[start_frame : start_frame + WINDOW_FRAMES]
    print(f"{clip_id}/{target}: window frames {start_frame}..{start_frame + len(window)}")

    last_bbox = None
    for i, frame_bgr in enumerate(window):
        h, w = frame_bgr.shape[:2]

        frame = Frame(
            timestamp_ms=float(i * 1000.0 / MOUTH_FPS),
            data=frame_bgr,
            metadata={"width": w, "height": h},
        )
        tracks = tracker.process_frame(frame)
        if tracks:
            md = tracks[0].metadata
            lip_bbox = md.get("lip_roi")
            face_bbox = tracks[0].bounding_box
            visible = md.get("visible")
            n_tracks = len(tracks)
            status = f"tracks={n_tracks} visible={visible} lip_roi={lip_bbox} face={face_bbox}"
            if lip_bbox is not None:
                last_bbox = lip_bbox
        else:
            status = "NO TRACKS"
        print(f"  frame {i:3d} (abs {start_frame+i}): {status}")

        if i % 6 == 0:
            annotated = frame_bgr.copy()
            if tracks and tracks[0].bounding_box:
                fx, fy, fw, fh = tracks[0].bounding_box
                cv2.rectangle(annotated, (int(fx), int(fy)), (int(fx + fw), int(fy + fh)), (0, 255, 0), 1)
            if tracks and tracks[0].metadata.get("lip_roi"):
                lx, ly, lw, lh = tracks[0].metadata["lip_roi"]
                cv2.rectangle(annotated, (int(lx), int(ly)), (int(lx + lw), int(ly + lh)), (0, 0, 255), 1)
            out_path = out_dir / f"{clip_id}_{target}_realtrack_frame{start_frame+i:04d}.png"
            cv2.imwrite(str(out_path), annotated)

    detector.close()
    print(f"\nlast known lip_bbox in window: {last_bbox}")

if __name__ == "__main__":
    main()
