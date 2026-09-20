
import sys
sys.path.insert(0, "src")
sys.path.insert(0, ".")

import numpy as np

import verify_lip_roi_real_crop as m
from bench._dolphin import load_dolphin, infer_separation
from onevoice.telemetry.quality import evaluate_pair
from onevoice.video.detectors.mediapipe_detector import MediaPipeFaceDetector

print("loading dolphin...")
model, device, stats = load_dolphin(device="cpu", warmup=False)
_detector = MediaPipeFaceDetector()

CASES = [("corpus_mix_08", "s1"), ("corpus_mix_04", "s1"), ("corpus_mix_04", "s2"), ("corpus_mix_01", "s1")]

for window_s in (1.5, 2.0):
    m.WINDOW_S = window_s
    m.WINDOW_SAMPLES = int(window_s * m.SAMPLE_RATE)
    m.WINDOW_FRAMES = int(window_s * m.MOUTH_FPS)
    print(f"\n===== window_s={window_s} =====")
    for clip_id, target in CASES:
        r = m.process_clip(clip_id, target, model, device, lambda: _detector)
        if r is None:
            print(f"{clip_id}/{target}: SKIPPED")
            continue
        line = (f"{clip_id}/{target}: new={r['new'][1]:+.2f}dB "
                f"ceiling={r['ceiling'][1]:+.2f}dB")

        if clip_id == "corpus_mix_04":
            other = "s2" if target == "s1" else "s1"
            corpus_dir = m.TT_ROOT / clip_id
            mixture = m._load_wav_float32(corpus_dir / "mix.wav")[: m.WINDOW_SAMPLES]
            ref_other = m._load_wav_float32(corpus_dir / f"{other}.wav")[: m.WINDOW_SAMPLES]
            import cv2
            cap = cv2.VideoCapture(str(m.MOUTH_CACHE_ROOT / clip_id / target / "temp_25fps.mp4"))
            frames = []
            for _ in range(m.WINDOW_FRAMES):
                ok, f = cap.read()
                if not ok:
                    break
                frames.append(f)
            cap.release()
            from onevoice.video.trackers.iou_tracker import IouFaceTracker, TrackerConfig
            tracker = IouFaceTracker(_detector, TrackerConfig(min_hits=1))
            mouth = m._new_pipeline_mouth_tensor(frames, tracker)
            est = infer_separation(model, mixture, mouth, device=device, fp16=False)
            k = min(len(est), m.WINDOW_SAMPLES)
            mo = evaluate_pair(ref_other[:k], est[:k], mixture[:k], m.SAMPLE_RATE)
            line += f"  [vs {other}: {mo['si_snr_improvement_db']:+.2f}dB]"
        print(line)
print("\ndone")
