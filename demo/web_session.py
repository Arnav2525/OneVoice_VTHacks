from __future__ import annotations

import base64
import json
import logging
import threading
import time
import webbrowser
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)
UI_ROOT = Path(__file__).resolve().parent / "web_ui"
MAX_BODY_BYTES = 2048
BODY_LIMITS = {"/api/speak": 4096}
STATIC_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/app.css": ("app.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/summary.js": ("summary.js", "text/javascript; charset=utf-8"),
    "/assets/arc_hero.png": ("assets/arc_hero.png", "image/png"),
}
STORY_ORIGIN = "http://127.0.0.1:4319"
STORY_ORIGINS = frozenset({STORY_ORIGIN, "http://localhost:4319"})


def _frame_info(runner: Any, session: Any) -> tuple[Any, dict[str, Any]]:
    frame = runner.frame()
    data = None if frame is None else frame.data
    shape: tuple[int, ...] = getattr(data, "shape", ())
    available = bool(
        runner.live
        and session.active
        and session.phase not in ("stopping", "interrupted")
        and len(shape) >= 2
        and shape[0] > 0
        and shape[1] > 0
    )
    metadata = {} if frame is None else frame.metadata
    width = int(shape[1]) if len(shape) >= 2 else int(metadata.get("width", 640))
    height = int(shape[0]) if len(shape) >= 2 else int(metadata.get("height", 480))
    return frame, {
        "available": available,
        "width": max(1, width),
        "height": max(1, height),
        "timestamp_ms": None if frame is None else frame.timestamp_ms,
    }


_NO_CAPTIONS = {"enabled": False, "status": "off", "error": None, "current": None}


def session_payload(runner: Any) -> dict[str, Any]:

    session = runner.state.snapshot()
    recording = asdict(runner.recorder.snapshot())
    recording["path"] = str(recording["path"]) if recording["path"] else None
    _, frame_info = _frame_info(runner, session)
    tracks = runner.selector.latest_tracks() if session.active else []
    captions = getattr(runner, "captions", None)
    return {
        "session": asdict(session),
        "tracks": [
            {
                "track_id": track.track_id,
                "bounding_box": [float(value) for value in track.bounding_box],
                "visible": bool(track.metadata.get("visible", True)),
                "person": runner.state.person_name(track.track_id),
            }
            for track in tracks
        ],
        "recording": recording,
        "record_busy": runner.record_busy,
        "busy": runner.busy,
        "synthetic": not runner.live,
        "identity": getattr(runner, "identity_status", lambda: {})(),
        "captions": captions.snapshot() if captions is not None else _NO_CAPTIONS,
        "recording_summary": (
            runner.recording_summaries.snapshot()
            if hasattr(runner, "recording_summaries")
            else None
        ),
        "frame": frame_info,
        "server_time_ms": time.monotonic() * 1000.0,
    }


class SessionHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    block_on_close = False

    def __init__(self, runner: Any, *, port: int = 0, ui_root: Path = UI_ROOT) -> None:
        self.runner = runner
        self.ui_root = ui_root
        self.action_lock = threading.Lock()
        self.summary_lock = threading.Lock()
        self.speak_lock = threading.Lock()
        self.closing = False
        self.cleanup_ok = True
        self._jpeg_lock = threading.Lock()
        self._jpeg_frame: Any = None
        self._jpeg: bytes | None = None
        super().__init__(("127.0.0.1", port), SessionRequestHandler)
        self.authority = f"127.0.0.1:{self.server_address[1]}"
        self.origin = f"http://{self.authority}"

    def get_request(self) -> Any:
        connection, address = super().get_request()
        connection.settimeout(3.0)
        return connection, address

    def jpeg(self) -> tuple[bytes, float] | None:
        frame, info = _frame_info(self.runner, self.runner.state.snapshot())
        if not info["available"]:
            return None
        with self._jpeg_lock:
            if frame is not self._jpeg_frame:
                import cv2

                ok, encoded = cv2.imencode(
                    ".jpg", frame.data, [cv2.IMWRITE_JPEG_QUALITY, 82]
                )
                self._jpeg = encoded.tobytes() if ok else None
                self._jpeg_frame = frame

            return (
                (self._jpeg, float(frame.timestamp_ms))
                if self._jpeg is not None
                else None
            )

    def request_quit(self) -> None:

        self.closing = True
        self.runner.stop()
        threading.Thread(
            target=self._quit, name="onevoice-web-shutdown", daemon=True
        ).start()

    def _quit(self) -> None:
        try:
            self.cleanup_ok = self.runner.close(timeout=5.0)
        except Exception:
            self.cleanup_ok = False
            logger.exception("Session cleanup failed while closing the browser UI")
        finally:
            self.shutdown()


class SessionRequestHandler(BaseHTTPRequestHandler):
    server: SessionHTTPServer

    def log_message(self, format: str, *args: Any) -> None:
        logger.debug("Browser UI: %s", format % args)

    def _reply(
        self,
        status: int,
        body: bytes = b"",
        content_type: str = "application/json",
        *,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.send_header("Cache-Control", "no-store, max-age=0")
        self.send_header("X-Content-Type-Options", "nosniff")
        if "Cross-Origin-Resource-Policy" not in (headers or {}):
            self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Permissions-Policy", "camera=(), microphone=()")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
            "img-src 'self' blob: data:; media-src blob:; connect-src 'self'; "
            "font-src 'self'; "
            "object-src 'none'; base-uri 'none'; frame-ancestors 'none'; "
            "form-action 'none'",
        )
        self.end_headers()
        if body:
            self.wfile.write(body)

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        self._reply(status, json.dumps(payload, allow_nan=False).encode("utf-8"))

    def _local_request(self) -> bool:

        hosts = self.headers.get_all("Host", [])
        origins = self.headers.get_all("Origin", [])
        if (
            hosts != [self.server.authority]
            or (origins and origins != [self.server.origin])
            or self.headers.get("Sec-Fetch-Site") == "cross-site"
        ):
            self._json(
                403, {"error": "This UI accepts only local same-origin requests."}
            )
            return False
        return True

    def _ping(self) -> None:
        if self.headers.get_all("Host", []) != [self.server.authority]:
            self._json(403, {"error": "This UI accepts only local requests."})
            return
        origin = self.headers.get("Origin")
        if origin not in STORY_ORIGINS:
            self._json(403, {"error": "Local story origin required."})
            return
        self._reply(
            200,
            b'{"app": "onevoice"}',
            headers={
                "Access-Control-Allow-Origin": origin,
                "Cross-Origin-Resource-Policy": "cross-origin",
                "Vary": "Origin",
            },
        )

    def do_GET(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path
        if path == "/api/ping":
            self._ping()
            return
        if not self._local_request():
            return
        try:
            if path == "/api/state":
                self._json(200, session_payload(self.server.runner))
            elif path == "/api/frame":
                jpeg = self.server.jpeg()
                if jpeg is None:
                    self._reply(204, content_type="image/jpeg")
                else:
                    body, timestamp_ms = jpeg
                    self._reply(
                        200,
                        body,
                        "image/jpeg",
                        headers={"X-Frame-Timestamp-Ms": str(timestamp_ms)},
                    )
            elif path in STATIC_FILES:
                filename, content_type = STATIC_FILES[path]
                self._reply(
                    200, (self.server.ui_root / filename).read_bytes(), content_type
                )
            else:
                self._json(404, {"error": "Not found"})
        except (BrokenPipeError, ConnectionResetError):
            pass
        except FileNotFoundError:
            self._json(404, {"error": "UI asset not found"})
        except Exception:
            logger.exception("Browser UI request failed")
            self._json(500, {"error": "Could not read session data. Stop and retry."})

    def _body(self) -> dict[str, Any] | None:

        lengths = self.headers.get_all("Content-Length", [])
        length = -1
        if len(lengths) == 1 and not self.headers.get("Transfer-Encoding"):
            try:
                length = int(lengths[0])
            except ValueError:
                pass
        body = b""
        limit = BODY_LIMITS.get(urlsplit(self.path).path, MAX_BODY_BYTES)
        if 0 < length <= limit + 1:
            try:
                body = self.rfile.read(length)
            except TimeoutError:
                self._json(400, {"error": "Request body timed out"})
                return None
        if not self._local_request():
            return None
        if self.headers.get("X-OneVoice-UI") != "1":
            self._json(403, {"error": "Missing UI request header"})
            return None
        if self.headers.get_content_type() != "application/json":
            self._json(415, {"error": "Use application/json"})
            return None
        if self.headers.get("Transfer-Encoding") or len(lengths) != 1:
            self._json(400, {"error": "A single Content-Length is required"})
            return None
        try:
            length = int(lengths[0])
        except ValueError:
            self._json(400, {"error": "Invalid Content-Length"})
            return None
        if not 0 < length <= limit:
            self._json(413, {"error": "Request body is too large or empty"})
            return None
        try:
            if len(body) != length:
                raise ValueError("Incomplete body")
            payload = json.loads(body.decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("Expected object")
            return payload
        except (ValueError, UnicodeError, TimeoutError):
            self._json(400, {"error": "A valid JSON object is required"})
            return None

    def do_POST(self) -> None:  # noqa: N802
        payload = self._body()
        if payload is None:
            return
        path = urlsplit(self.path).path
        if path == "/api/summarize":
            self._summarize()
            return
        if path == "/api/speak":
            self._speak(payload)
            return
        if path not in ("/api/action", "/api/quit"):
            self._json(404, {"error": "Not found"})
            return
        try:
            with self.server.action_lock:
                if self.server.closing:
                    self._json(409, {"error": "The session is closing"})
                    return
                if path == "/api/quit":
                    self._json(200, {"ok": True})
                    self.server.request_quit()
                    return
                runner = self.server.runner
                action = payload.get("action")
                if action == "start":
                    runner.start()
                elif action == "stop":
                    runner.stop()
                elif action == "record":
                    runner.toggle_recording()
                elif action == "captions":
                    runner.toggle_captions()
                elif action in ("select", "clear"):
                    track_id = payload.get("track_id") if action == "select" else None
                    if action == "select" and (
                        not isinstance(track_id, str)
                        or not track_id
                        or len(track_id) > 256
                    ):
                        self._json(400, {"error": "Select requires a track_id"})
                        return
                    if not runner.state.select(track_id):
                        self._json(409, {"error": "That person is no longer available"})
                        return
                else:
                    self._json(400, {"error": "Unknown action"})
                    return
                self._json(200, session_payload(runner))
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception:
            logger.exception("Browser UI action failed")
            self._json(500, {"error": "The action failed. Stop and retry."})

    def _summarize(self) -> None:
        from demo.summary import summarize

        if not self.server.summary_lock.acquire(blocking=False):
            self._json(409, {"error": "A summary is already in progress."})
            return
        try:
            if self.server.closing:
                self._json(409, {"error": "The session is closing."})
                return
            captions = getattr(self.server.runner, "captions", None)
            transcript = captions.transcript_text() if captions is not None else ""
            if not transcript.strip():
                self._json(
                    409,
                    {
                        "error": "No captions yet. Turn on Captions and let the "
                        "selected person speak first."
                    },
                )
                return
            self._json(200, summarize(transcript))
        except ValueError as exc:
            self._json(400, {"error": str(exc)})
        except RuntimeError as exc:
            self._json(502, {"error": str(exc)})
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception:
            logger.exception("Summary failed")
            self._json(500, {"error": "Could not summarize. Please retry."})
        finally:
            self.server.summary_lock.release()

    def _speak(self, payload: dict[str, Any]) -> None:
        from demo.speech import SAMPLE_RATE, synthesize, to_wav

        if not self.server.speak_lock.acquire(blocking=False):
            self._json(409, {"error": "Speech is already being prepared."})
            return
        try:
            if self.server.closing:
                self._json(409, {"error": "The session is closing."})
                return
            text = payload.get("text")
            if not isinstance(text, str):
                raise ValueError("Send the text to speak.")
            wav = to_wav(synthesize(text))
            self._json(
                200,
                {
                    "audio": base64.b64encode(wav).decode("ascii"),
                    "sample_rate": SAMPLE_RATE,
                },
            )
        except ValueError as exc:
            self._json(400, {"error": str(exc)})
        except RuntimeError as exc:
            self._json(502, {"error": str(exc)})
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception:
            logger.exception("Speech failed")
            self._json(500, {"error": "Could not prepare speech. Please retry."})
        finally:
            self.server.speak_lock.release()


def run_web(runner: Any, *, open_browser: bool = True, port: int = 0) -> int:

    server = SessionHTTPServer(runner, port=port)
    url = f"{server.origin}/"
    print(f"OneVoice: {url}\nPress Start listening in the page. Ctrl+C exits.")
    if open_browser:
        try:
            webbrowser.open(url, new=2)
        except Exception:
            logger.exception("Could not open the browser; use the URL above")
    try:
        server.serve_forever(poll_interval=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        with server.action_lock:
            server.closing = True
        try:
            server.cleanup_ok = runner.close(timeout=5.0)
        except Exception:
            server.cleanup_ok = False
            logger.exception("Session cleanup failed")
        server.server_close()
    if not server.cleanup_ok:
        logger.error("Some session resources could not be released; see errors above")
    return 0 if server.cleanup_ok else 1
