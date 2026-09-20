

from __future__ import annotations

import math
from typing import Any, cast

import cv2
import numpy as np

from demo._render import draw_corner_brackets
from demo.session_state import SessionView
from demo.tap_selection import hit_test

Rect = tuple[int, int, int, int]
INK = (240, 242, 242)
MUTED = (184, 174, 163)
ACCENT = (136, 225, 123)
BLUE = (238, 181, 106)
AMBER = (95, 194, 245)

def text(
    canvas: Any,
    value: str,
    xy: tuple[int, int],
    scale: float = 0.55,
    color: tuple[int, int, int] = INK,
    weight: int = 1,
) -> None:
    cv2.putText(
        canvas, value, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, weight, cv2.LINE_AA
    )

def wrap(
    canvas: Any,
    value: str,
    x: int,
    y: int,
    width: int,
    scale: float = 0.55,
    color: tuple[int, int, int] = MUTED,
    line_height: int = 25,
    max_lines: int = 6,
) -> int:

    words = value.split()
    lines: list[str] = []
    line = ""
    for word in words:
        candidate = f"{line} {word}".strip()
        if (
            line
            and cv2.getTextSize(candidate, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0]
            > width
        ):
            lines.append(line)
            line = word
        else:
            line = candidate
    if line:
        lines.append(line)
    for i, line in enumerate(lines[:max_lines]):
        shortened = i == max_lines - 1 and len(lines) > max_lines
        while (
            line
            and cv2.getTextSize(
                line + ("..." if shortened else ""),
                cv2.FONT_HERSHEY_SIMPLEX,
                scale,
                1,
            )[0][0]
            > width
        ):
            line = line[:-1]
            shortened = True
        text(canvas, line + ("..." if shortened else ""), (x, y), scale, color)
        y += line_height
    return y

def inside(x: float, y: float, rect: Rect) -> bool:
    rx, ry, rw, rh = rect
    return rx <= x < rx + rw and ry <= y < ry + rh

def face_at(
    x: float, y: float, rect: Rect, native: tuple[int, int], tracks: list[Any]
) -> str | None:
    if not inside(x, y, rect):
        return None
    rx, ry, rw, rh = rect
    return cast(
        str | None,
        hit_test(
            [t for t in tracks if t.metadata.get("visible", True)],
            (x - rx) * native[0] / rw,
            (y - ry) * native[1] / rh,
        ),
    )

def button(
    canvas: Any, rect: Rect, label: str, *, enabled: bool = True, primary: bool = False
) -> None:
    x, y, w, h = rect
    fill = ACCENT if primary and enabled else (59, 52, 45)
    foreground = (20, 31, 27) if primary and enabled else INK if enabled else MUTED
    cv2.rectangle(canvas, (x, y), (x + w, y + h), fill, -1)
    label_width = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.53, 1)[0][0]
    text(canvas, label, (x + (w - label_width) // 2, y + h // 2 + 6), 0.53, foreground)

def meter(
    canvas: Any,
    x: int,
    y: int,
    width: int,
    label: str,
    value: float,
    color: tuple[int, int, int],
) -> None:
    text(canvas, label, (x, y), 0.48, MUTED)
    db = max(-60.0, 20 * math.log10(max(value, 1e-6)))
    fraction = (db + 60) / 60
    cv2.rectangle(canvas, (x, y + 12), (x + width, y + 23), (70, 61, 53), -1)
    if fraction > 0:
        cv2.rectangle(
            canvas, (x, y + 12), (x + int(width * fraction), y + 23), color, -1
        )

def preview_picture(tracks: list[Any]) -> np.ndarray:
    canvas: np.ndarray = np.full((480, 640, 3), (44, 35, 28), dtype=np.uint8)
    colors = ((116, 105, 78), (125, 96, 125), (97, 127, 83))
    for i, track in enumerate(tracks):
        if not track.metadata.get("visible", True):
            continue
        x, y, w, h = (int(v) for v in track.bounding_box)
        cx = x + w // 2
        cv2.rectangle(
            canvas, (x - 8, y - 14), (x + w + 8, y + h + 15), (65, 52, 43), -1
        )
        cv2.circle(canvas, (cx, y + 72), 44, colors[i % 3], -1, cv2.LINE_AA)
        cv2.ellipse(
            canvas, (cx, y + 190), (60, 65), 0, 180, 360, colors[i % 3], -1, cv2.LINE_AA
        )
        cv2.circle(canvas, (cx - 15, y + 66), 3, INK, -1)
        cv2.circle(canvas, (cx + 15, y + 66), 3, INK, -1)
        cv2.line(canvas, (cx - 12, y + 90), (cx + 12, y + 90), INK, 2)
    text(canvas, "SIMULATED PEOPLE - NO CAMERA OR MICROPHONE", (29, 439), 0.49, MUTED)
    return canvas

def render_session(
    view: SessionView,
    frame: Any,
    tracks: list[Any],
    names: Any,
    recording: Any,
    *,
    size: tuple[int, int] = (1280, 800),
    synthetic: bool = True,
    record_busy: bool = False,
) -> tuple[np.ndarray, dict[str, Rect], Rect, tuple[int, int]]:
    w, h = max(960, size[0]), max(640, size[1])
    canvas: np.ndarray = np.full((h, w, 3), (29, 23, 18), dtype=np.uint8)
    text(canvas, "ONEVOICE", (28, 43), 0.9, INK, 2)
    text(canvas, "Your listening session", (29, 70), 0.47, MUTED)
    mode = (
        "CPU PREVIEW"
        if synthetic
        else "CAPTURE PREVIEW" if view.mode != "live" else "LIVE SESSION"
    )
    mode_color = BLUE if view.mode != "live" else ACCENT
    text(canvas, mode, (w - 272, 43), 0.55, mode_color)
    text(canvas, "One selected voice at a time", (w - 272, 68), 0.42, MUTED)
    cv2.line(canvas, (24, 87), (w - 24, 87), (49, 57, 65), 1)

    panel_x, panel_w = w - 305, 281
    area = (24, 106, w - 353, h - 225)
    ax, ay, aw, ah = area
    cv2.rectangle(canvas, (ax, ay), (ax + aw, ay + ah), (38, 30, 24), -1)
    native = (640, 480)
    image = frame.data if frame is not None else None
    active_tracks = tracks if view.active and view.phase != "stopping" else []
    if image is None and synthetic and active_tracks:
        image = preview_picture(active_tracks)
    rect = area
    if image is not None:
        ih, iw = image.shape[:2]
        native = (iw, ih)
        scale = min(aw / iw, ah / ih)
        dw, dh = int(iw * scale), int(ih * scale)
        dx, dy = ax + (aw - dw) // 2, ay + (ah - dh) // 2
        rect = (dx, dy, dw, dh)
        picture = cv2.resize(image, (dw, dh))
        if not view.active or view.phase in ("interrupted", "stopping"):
            picture = (picture * 0.4).astype(np.uint8)
        canvas[dy : dy + dh, dx : dx + dw] = picture
        for track in active_tracks:
            if not track.metadata.get("visible", True):
                continue
            tx, ty, tw, th = track.bounding_box
            bx, by, bw, bh = (
                int(dx + tx * scale),
                int(dy + ty * scale),
                int(tw * scale),
                int(th * scale),
            )
            selected = view.selected_id == track.track_id
            color = (
                ACCENT
                if selected and view.mode == "live"
                else BLUE if selected else MUTED
            )
            draw_corner_brackets(canvas, bx, by, bw, bh, color, 3 if selected else 1)
            label = names(track.track_id)
            text(canvas, label, (bx + 3, max(dy + 20, by - 10)), 0.53, color)
    else:
        message = (
            "Camera preview appears after Start"
            if view.phase in ("ready", "stopped", "error")
            else "Waiting for video..."
        )
        wrap(canvas, message, ax + 25, ay + ah // 2, aw - 50, 0.62)
    text(canvas, f"{view.face_count} people visible", (ax + 10, h - 101), 0.47, MUTED)
    if synthetic:
        text(
            canvas,
            "Preview uses synthetic people and audio",
            (ax + 190, h - 101),
            0.43,
            BLUE,
        )

    cv2.rectangle(canvas, (panel_x, 106), (w - 24, h - 119), (46, 37, 30), -1)
    sx, sw = panel_x + 20, panel_w - 40
    state_color = (
        ACCENT
        if view.phase == "isolating"
        else (
            AMBER
            if view.phase in ("target_lost", "error", "interrupted", "unavailable")
            else BLUE
        )
    )
    badge = (
        "HEARING" if view.phase == "isolating" else view.phase.replace("_", " ").upper()
    )
    text(canvas, badge, (sx, 137), 0.44, state_color)
    meter_y = h - 280
    y = wrap(canvas, view.title, sx, 178, sw, 0.7, INK, 31, max_lines=3)
    wrap(
        canvas,
        view.detail,
        sx,
        y + 14,
        sw,
        0.48,
        MUTED,
        23,
        max_lines=max(1, (meter_y - y - 36) // 23),
    )
    input_label = "SIMULATED INPUT" if synthetic else "MICROPHONE INPUT"
    meter(canvas, sx, meter_y, sw, input_label, view.input_level, BLUE)
    meter(canvas, sx, meter_y + 63, sw, "ISOLATED OUTPUT", view.output_level, ACCENT)
    if view.elapsed_s:
        minutes, seconds = divmod(int(view.elapsed_s), 60)
        text(canvas, f"Session {minutes:02}:{seconds:02}", (sx, h - 141), 0.47, MUTED)

    controls = {
        "session": (24, h - 80, 188, 46),
        "record": (225, h - 80, 184, 46),
        "clear": (422, h - 80, 167, 46),
    }
    start_label = (
        "Stopping..."
        if view.phase == "stopping"
        else (
            "Cancel start"
            if view.active and view.phase == "starting"
            else "Stop session" if view.active else "Start listening"
        )
    )
    button(
        canvas,
        controls["session"],
        start_label,
        enabled=view.phase != "stopping",
        primary=not view.active,
    )
    can_record = view.phase in ("listening", "focusing", "isolating", "target_lost")
    record_label = (
        "Working..."
        if record_busy
        else "Stop recording" if recording.active else "Record clip"
    )
    button(
        canvas,
        controls["record"],
        record_label,
        enabled=(can_record or recording.active) and not record_busy,
    )
    button(
        canvas,
        controls["clear"],
        "Clear selection",
        enabled=view.active and view.selected_id is not None,
    )
    if recording.active:
        mins, secs = divmod(int(recording.elapsed_s), 60)
        cv2.circle(canvas, (w - 256, h - 58), 5, (95, 95, 245), -1)
        text(
            canvas, f"REC {mins:02}:{secs:02}", (w - 240, h - 51), 0.6, (120, 130, 255)
        )
    elif recording.error:
        text(canvas, "Recording failed - see console", (w - 305, h - 52), 0.42, AMBER)
    elif recording.path and not record_busy:
        text(canvas, "Clip saved", (w - 240, h - 52), 0.55, ACCENT)
    text(
        canvas,
        "SPACE start/stop    R record    C clear    F fullscreen    Q quit",
        (26, h - 12),
        0.4,
        MUTED,
    )
    return canvas, controls, rect, native
