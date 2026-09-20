from __future__ import annotations

import base64
import json
import os
import queue
import threading
import time
from collections import deque
from urllib.parse import urlencode

import numpy as np


class RealtimeStream:
    def __init__(self, callback, status, settings):
        self.callback = callback
        self.status = status
        self.settings = settings
        self.queue = queue.Queue(maxsize=100)
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def write(self, chunk):
        if chunk.sample_rate != 16000 or chunk.channels != 1:
            self.status("error", "Live captions require 16 kHz mono audio.")
            self.stop()
            return
        samples = np.asarray(chunk.data, dtype=np.float32)
        pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes()
        try:
            self.queue.put_nowait(pcm)
        except queue.Full:
            self.status("error", "Captions fell behind. Toggle Captions to reconnect.")
            self.stop()

    def stop(self):
        self.stopped.set()

    def run(self):
        try:
            import websocket
        except ImportError:
            self.status("error", "Install websocket-client, then restart the app.")
            return

        key = os.environ.get("ELEVENLABS_API_KEY", "").strip()
        if not key:
            self.status(
                "error", "Add ELEVENLABS_API_KEY to .env, then restart the app."
            )
            return
        params = urlencode(
            {
                "model_id": self.settings.get("model", "scribe_v2_realtime"),
                "audio_format": "pcm_16000",
                "commit_strategy": "vad",
                "vad_silence_threshold_secs": 0.8,
            }
        )
        for attempt in range(3):
            if self.stopped.is_set():
                return
            ws = None
            try:
                self.status("loading", None)
                ws = websocket.create_connection(
                    "wss://api.elevenlabs.io/v1/speech-to-text/realtime?" + params,
                    header={"xi-api-key": key},
                    timeout=3,
                )
                ws.settimeout(0.02)
                pending = bytearray()
                while not self.stopped.is_set():
                    while len(pending) < 32000:
                        try:
                            pending.extend(self.queue.get_nowait())
                        except queue.Empty:
                            break
                    while len(pending) >= 6400 and not self.stopped.is_set():
                        packet = bytes(pending[:6400])
                        del pending[:6400]
                        ws.settimeout(1)
                        ws.send(
                            json.dumps(
                                {
                                    "message_type": "input_audio_chunk",
                                    "audio_base_64": base64.b64encode(packet).decode(),
                                    "sample_rate": 16000,
                                }
                            )
                        )
                        ws.settimeout(0.02)
                    try:
                        message = json.loads(ws.recv())
                    except websocket.WebSocketTimeoutException:
                        continue
                    kind = message.get("message_type", "")
                    if kind == "session_started":
                        self.status("ready", None)
                    elif kind in ("partial_transcript", "committed_transcript"):
                        self.callback(
                            message.get("text", ""), kind == "committed_transcript"
                        )
                    elif (
                        "error" in message
                        or "error" in kind
                        or kind in ("quota_exceeded", "rate_limited")
                    ):
                        self.status(
                            "error",
                            "ElevenLabs captions unavailable ("
                            + kind
                            + "). Check your key and credits, then toggle Captions.",
                        )
                        return
                while not self.queue.empty():
                    pending.extend(self.queue.get_nowait())
                ws.settimeout(1)
                ws.send(
                    json.dumps(
                        {
                            "message_type": "input_audio_chunk",
                            "audio_base_64": base64.b64encode(
                                pending or bytes(3200)
                            ).decode(),
                            "sample_rate": 16000,
                            "commit": True,
                        }
                    )
                )
                ws.settimeout(0.05)
                deadline = time.monotonic() + 0.8
                while time.monotonic() < deadline:
                    try:
                        message = json.loads(ws.recv())
                    except websocket.WebSocketTimeoutException:
                        continue
                    if message.get("message_type") == "committed_transcript":
                        self.callback(message.get("text", ""), True)
                        break
            except Exception:
                if not self.stopped.is_set():
                    self.status("reconnecting", "Caption connection interrupted.")
            finally:
                if ws is not None:
                    ws.close()
            while not self.queue.empty():
                try:
                    self.queue.get_nowait()
                except queue.Empty:
                    break
            if self.stopped.wait(1 + attempt):
                return
        self.status(
            "error",
            "Could not connect to ElevenLabs. Check internet and API access, "
            "then toggle Captions.",
        )


class ElevenLabsCaptions:
    def __init__(self, state, settings=None, stream_factory=RealtimeStream):
        self.state = state
        self.settings = settings or {}
        self.factory = stream_factory
        self.lock = threading.RLock()
        self.enabled = False
        self.status = "off"
        self.error = None
        self.current = None
        self.stream = None
        self.key = None
        self.generation = 0
        self.retired = deque(maxlen=64)
        self.lines = deque(maxlen=300)
        self.revision = 0

    def toggle(self):
        with self.lock:
            self.enabled = not self.enabled
            self.retire_current_sink()
            self.status = "waiting" if self.enabled else "off"
            self.error = None

    def retire_current_sink(self):
        with self.lock:
            self.retired.append(self.generation)
            self.generation += 1
            if self.stream is not None:
                self.stream.stop()
            self.stream = None
            self.key = None
            self.current = None

    def on_output(self, chunk, allowed):
        selected = self.state.selection()
        with self.lock:
            if not self.enabled:
                return
            if self.key is not None and self.key != selected:
                self.retire_current_sink()
                self.status, self.error = "waiting", None
            if not allowed or selected[0] is None:
                if self.stream is not None:
                    self.retire_current_sink()
                if self.status != "error":
                    self.status = "waiting"
                return
            if chunk.metadata.get("target_track_id") != selected[0]:
                return
            if chunk.metadata.get("ui_selection_epoch") != selected[1]:
                return
            if self.status == "error":
                return
            if self.stream is None:
                self.key = selected
                generation = self.generation
                self.stream = self.factory(
                    lambda text, final: self.receive(generation, selected, text, final),
                    lambda status, error: self.set_status(generation, status, error),
                    self.settings,
                )
            self.stream.write(chunk)

    def set_status(self, generation, status, error):
        with self.lock:
            if self.enabled and generation == self.generation:
                self.status, self.error = status, error

    def receive(self, generation, selected, text, final):
        with self.lock:
            if selected != self.state.selection() or not text.strip():
                return
            if generation != self.generation:
                if final and generation in self.retired:
                    self.lines.append(
                        (self.state.person_name(selected[0]), text.strip())
                    )
                return
            if not self.enabled:
                return
            self.revision += 1
            self.current = {
                "track_id": selected[0],
                "text": text.strip(),
                "timestamp_ms": time.monotonic() * 1000,
                "revision": self.revision,
                "final": final,
            }
            if final:
                self.lines.append((self.state.person_name(selected[0]), text.strip()))

    def snapshot(self):
        with self.lock:
            if self.key is not None and self.key != self.state.selection():
                self.retire_current_sink()
                self.status = "waiting" if self.enabled else "off"
            return {
                "enabled": self.enabled,
                "status": self.status,
                "error": self.error,
                "current": self.current,
            }

    def transcript_text(self, max_chars=12000):
        with self.lock:
            return "\n".join(f"{speaker}: {text}" for speaker, text in self.lines)[
                -max_chars:
            ]

    def clear_transcript(self):
        with self.lock:
            self.retire_current_sink()
            self.lines.clear()
            self.retired.clear()
            self.status = "waiting" if self.enabled else "off"
            self.error = None

    def close(self):
        with self.lock:
            self.enabled = False
            self.retire_current_sink()
            self.status = "off"
