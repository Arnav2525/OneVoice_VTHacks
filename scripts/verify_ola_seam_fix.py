

from __future__ import annotations

import sys
import wave
import array
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from onevoice.core.models.audio_chunk import AudioChunk
from onevoice.core.models.speaker_track import SpeakerTrack
from onevoice.core.models.target_selection import TargetSelection
from onevoice.separation.adapters.base import AdapterContext
from onevoice.separation.adapters.dolphin_adapter import DolphinAdapter
from onevoice.separation.config import SeparationConfig
from onevoice.separation.dolphin_loader import load_dolphin_model

def load_wav_float32(path: str) -> tuple[np.ndarray, int]:
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        n = w.getnframes()
        raw = w.readframes(n)
    samples = np.asarray(array.array("h", raw), dtype=np.float32) / 32768.0
    return samples, sr

def main() -> None:
    wav_path = "runs/live_dolphin_recording_v9/mixture_before.wav"
    samples, sr = load_wav_float32(wav_path)
    print(f"loaded {wav_path}: {len(samples)} samples @ {sr} Hz ({len(samples)/sr:.1f}s)")

    config = SeparationConfig(
        name="dolphin",
        device="cpu",
        sample_rate=sr,
        chunk_size=320,
        params={"window_s": 1.5, "hop_s": 0.75, "auto_gain": True},
    )
    print("loading real Dolphin model (fp32, cpu)...")
    model = load_dolphin_model(config, "cpu")
    adapter = DolphinAdapter(config)
    ctx = AdapterContext(device="cpu", config=config)

    target = TargetSelection(0.0, SpeakerTrack("spk-1", (0, 0, 10, 10), 0.9, metadata={}))

    rng = np.random.default_rng(42)
    _patches = [
        [0.0] * 64,
        [0.8] * 64,
        list(rng.uniform(-1.0, 1.0, size=64)),
        [((-1.0) ** k) * 0.5 for k in range(64)],
    ]

    def lip_patch_for(chunk_index: int) -> list:
        regime = (chunk_index // 50) % len(_patches)
        return _patches[regime]

    chunk_samples = 320
    output_chunks: list[np.ndarray] = []
    n_chunks = len(samples) // chunk_samples
    print(f"feeding {n_chunks} chunks of {chunk_samples} samples through the adapter (varying lip_patch)...")

    for i in range(n_chunks):
        block = samples[i * chunk_samples : (i + 1) * chunk_samples]
        lip_patch = lip_patch_for(i)
        chunk = AudioChunk(
            timestamp_ms=float(i * chunk_samples * 1000.0 / sr),
            data=array.array("f", block.tolist()),
            sample_rate=sr,
            channels=1,
            metadata={},
        )
        speaker = SpeakerTrack("spk-1", (0, 0, 10, 10), 0.9, metadata={"lip_patch": lip_patch})
        t = TargetSelection(chunk.timestamp_ms, speaker)
        backend_input = adapter.to_backend(model, chunk, t, [speaker], ctx)
        out = adapter.infer(model, backend_input, ctx)
        output_chunks.append(np.asarray(out, dtype=np.float32))
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{n_chunks} chunks processed")

    try:
        adapter.flush_pending("spk-1", timeout_s=180.0)
    except TimeoutError:
        print("WARNING: final trailing window did not finish within 180s "
              "(CPU contention) -- proceeding with analysis on chunks gathered so far.")

    remaining_chunk = AudioChunk(
        timestamp_ms=float(n_chunks * chunk_samples * 1000.0 / sr),
        data=array.array("f", [0.0] * chunk_samples),
        sample_rate=sr,
        channels=1,
        metadata={},
    )
    speaker = SpeakerTrack("spk-1", (0, 0, 10, 10), 0.9, metadata={"lip_patch": lip_patch})
    t = TargetSelection(remaining_chunk.timestamp_ms, speaker)
    backend_input = adapter.to_backend(model, remaining_chunk, t, [speaker], ctx)
    out = adapter.infer(model, backend_input, ctx)
    output_chunks.append(np.asarray(out, dtype=np.float32))
    adapter.shutdown()

    full_output = np.concatenate(output_chunks)
    print(f"\ntotal output samples: {len(full_output)} ({len(full_output)/sr:.1f}s)")

    clipped = np.sum(np.abs(full_output) >= 0.999)
    print(f"samples at/near clip ceiling (|x|>=0.999): {clipped} / {len(full_output)}")

    diffs = np.abs(np.diff(full_output))
    typical_diff = np.median(diffs[diffs > 0]) if np.any(diffs > 0) else 0.0
    big_jump_threshold = max(0.05, typical_diff * 50)
    big_jumps = np.where(diffs > big_jump_threshold)[0]
    print(f"typical sample-to-sample step: {typical_diff:.6g}")
    print(f"big-jump threshold used: {big_jump_threshold:.6g}")
    print(f"big jumps found: {len(big_jumps)}")

    hop_samples = int(round(0.75 * sr))
    if len(big_jumps) > 0:
        near_hop_boundary = 0
        for idx in big_jumps:
            phase = idx % hop_samples
            if phase < 50 or phase > hop_samples - 50:
                near_hop_boundary += 1
        print(f"big jumps within 50 samples of a hop boundary: {near_hop_boundary}/{len(big_jumps)}")
        print("sample big-jump indices (first 10):", big_jumps[:10].tolist())
    else:
        print("no big jumps detected anywhere in the reconstructed output.")

    print(f"\nmax_abs={np.abs(full_output).max():.4f} mean_abs={np.abs(full_output).mean():.6f}")

    out_path = "runs/ola_seam_verification_output_varying_lip_patch.wav"
    Path("runs").mkdir(exist_ok=True)
    int16 = np.clip(full_output * 32767.0, -32768, 32767).astype(np.int16)
    with wave.open(out_path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(int16.tobytes())
    print(f"saved reconstructed output to {out_path}")

if __name__ == "__main__":
    main()
