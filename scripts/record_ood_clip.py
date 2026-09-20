

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from onevoice.audio.io import MicrophoneSource, WavFileSink
from onevoice.video.capture import WebcamSource

def record(
    duration_s: float,
    output_dir: Path,
    device_index: int = 0,
    width: int = 640,
    height: int = 480,
    fps: float = 25.0,
    sample_rate: int = 16_000,
    mic_device: int | None = None,
) -> tuple[Path, Path]:
    import cv2

    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / "target_video.mp4"
    audio_path = output_dir / "target_audio.wav"

    video_source = WebcamSource(device_index=device_index, width=width, height=height, fps=fps)
    mic_source = MicrophoneSource(sample_rate=sample_rate, channels=1, chunk_samples=320, device=mic_device)
    audio_sink = WavFileSink(audio_path, sample_rate, 1)

    video_source.start()
    mic_source.start()
    audio_sink.start()

    frames: list = []
    stop_flag = threading.Event()

    def video_loop() -> None:
        while not stop_flag.is_set():
            try:
                frame = video_source.read()
            except Exception:  # noqa: BLE001 - camera hiccup shouldn't kill the whole recording
                continue
            frames.append(frame.data)

    def audio_loop() -> None:
        deadline = time.monotonic() + duration_s
        while time.monotonic() < deadline:
            chunk = mic_source.read()
            audio_sink.write(chunk)
        stop_flag.set()

    video_thread = threading.Thread(target=video_loop, daemon=True)
    audio_thread = threading.Thread(target=audio_loop, daemon=True)
    print(f"recording {duration_s:.1f}s -- talk naturally, move a bit, look away sometimes...")
    video_thread.start()
    audio_thread.start()
    audio_thread.join()
    video_thread.join(timeout=1.0)

    video_source.stop()
    mic_source.stop()
    audio_sink.stop()

    if not frames:
        raise RuntimeError("no video frames captured -- check webcam device index")

    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for frame_bgr in frames:
        writer.write(frame_bgr)
    writer.release()

    print(f"wrote {len(frames)} frames -> {video_path}")
    print(f"wrote audio -> {audio_path}")
    return video_path, audio_path

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=15.0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--mic-device", type=int, default=None)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--sample-rate", type=int, default=16_000)
    args = parser.parse_args(argv)

    record(
        duration_s=args.duration,
        output_dir=Path(args.output_dir),
        device_index=args.device_index,
        width=args.width,
        height=args.height,
        fps=args.fps,
        sample_rate=args.sample_rate,
        mic_device=args.mic_device,
    )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
