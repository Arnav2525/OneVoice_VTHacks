from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

API_ROOT = "https://api.elevenlabs.io/v1/text-to-speech"
SAMPLE_RATE = 16_000
DEFAULT_MODEL = "eleven_flash_v2_5"
DEFAULT_VOICE = "21m00Tcm4TlvDq8ikWAM"
MAX_CHARS = 500
TIMEOUT_S = 20


class SpeechError(RuntimeError):
    pass


def synthesize(
    text: str,
    *,
    voice_id: str | None = None,
    model_id: str | None = None,
) -> bytes:
    text = text.strip()
    if not text:
        raise SpeechError("There is nothing to speak.")
    if len(text) > MAX_CHARS:
        raise SpeechError(f"Keep spoken text under {MAX_CHARS} characters.")
    key = os.environ.get("ELEVENLABS_API_KEY", "")
    if not key:
        raise SpeechError("Set ELEVENLABS_API_KEY on the server, then retry.")
    voice = voice_id or os.environ.get("ELEVENLABS_VOICE_ID") or DEFAULT_VOICE
    model = model_id or os.environ.get("ONEVOICE_ELEVENLABS_MODEL") or DEFAULT_MODEL
    request = urllib.request.Request(
        f"{API_ROOT}/{voice}?output_format=pcm_{SAMPLE_RATE}",
        data=json.dumps({"text": text, "model_id": model}).encode("utf-8"),
        headers={
            "xi-api-key": key,
            "Content-Type": "application/json",
            "Accept": "application/octet-stream",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
            pcm = response.read()
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            raise SpeechError("ElevenLabs rejected the API key.") from None
        if exc.code == 429:
            raise SpeechError("ElevenLabs quota or rate limit reached.") from None
        if exc.code in (400, 404, 422):
            raise SpeechError("ElevenLabs rejected the voice, model or text.") from None
        raise SpeechError(f"ElevenLabs request failed (HTTP {exc.code}).") from None
    except (urllib.error.URLError, TimeoutError):
        raise SpeechError("Could not reach ElevenLabs.") from None
    if len(pcm) < 2:
        raise SpeechError("ElevenLabs returned no audio.")
    return pcm[: len(pcm) - len(pcm) % 2]


def play(pcm: bytes, device: int | str | None = None) -> None:
    import numpy as np

    from demo.devices import resolve_audio_device

    try:
        import sounddevice as sd
    except (ImportError, OSError) as exc:
        raise SpeechError("Playback needs sounddevice and PortAudio.") from exc
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
    output = resolve_audio_device(device, kind="output")
    sd.play(samples, SAMPLE_RATE, device=output)
    sd.wait()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Speak text with ElevenLabs.")
    parser.add_argument("text", help="short text to speak")
    parser.add_argument("--output-device", default=None)
    parser.add_argument(
        "--save", default=None, help="write raw 16 kHz mono PCM here instead of playing"
    )
    args = parser.parse_args(argv)
    try:
        pcm = synthesize(args.text)
        if args.save:
            Path(args.save).write_bytes(pcm)
            print(f"Saved {len(pcm) / 2 / SAMPLE_RATE:.1f}s of audio to {args.save}")
        else:
            play(pcm, args.output_device)
    except (SpeechError, ValueError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
