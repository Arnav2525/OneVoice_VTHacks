

from __future__ import annotations

import math
import time

import cv2
import numpy as np

def filled_rounded_rect(
    canvas: np.ndarray,
    top_left: tuple[int, int],
    bottom_right: tuple[int, int],
    color: tuple[int, int, int],
    radius: int,
) -> None:

    x1, y1 = top_left
    x2, y2 = bottom_right
    radius = max(0, min(radius, (x2 - x1) // 2, (y2 - y1) // 2))
    if radius == 0:
        cv2.rectangle(canvas, top_left, bottom_right, color, -1)
        return
    cv2.rectangle(canvas, (x1 + radius, y1), (x2 - radius, y2), color, -1)
    cv2.rectangle(canvas, (x1, y1 + radius), (x2, y2 - radius), color, -1)
    for corner_x, corner_y in (
        (x1 + radius, y1 + radius),
        (x2 - radius, y1 + radius),
        (x1 + radius, y2 - radius),
        (x2 - radius, y2 - radius),
    ):
        cv2.circle(canvas, (corner_x, corner_y), radius, color, -1, cv2.LINE_AA)

def label_with_background(
    canvas: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
    font_scale: float = 0.6,
    thickness: int = 1,
) -> None:

    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = origin
    pad = 7
    top_left = (x - pad, y - th - pad)
    bottom_right = (x + tw + pad, y + baseline + pad)
    overlay = canvas.copy()
    filled_rounded_rect(overlay, top_left, bottom_right, (20, 20, 20), radius=pad + 3)
    cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0, dst=canvas)
    cv2.putText(canvas, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)

def display_size(window_name: str, fallback: tuple[int, int]) -> tuple[int, int]:

    try:
        _, _, w, h = cv2.getWindowImageRect(window_name)
        if w > 0 and h > 0:
            return w, h
    except Exception:  # noqa: BLE001 - window not ready yet
        pass
    return fallback

def draw_vu_meter(
    canvas: np.ndarray,
    top_left: tuple[int, int],
    size: tuple[int, int],
    rms: float,
    peak: float,
    color: tuple[int, int, int],
    bg_color: tuple[int, int, int] = (35, 35, 35),
) -> None:

    x, y = top_left
    w, h = size
    rms = min(1.0, max(0.0, rms))
    peak = min(1.0, max(0.0, peak))

    filled_rounded_rect(canvas, (x, y), (x + w, y + h), bg_color, radius=h // 2)
    fill_w = int(w * rms)
    if fill_w > 0:
        filled_rounded_rect(canvas, (x, y), (x + fill_w, y + h), color, radius=h // 2)

    peak_x = x + int(w * peak)
    peak_x = min(max(peak_x, x + 1), x + w - 1)
    cv2.line(canvas, (peak_x, y - 2), (peak_x, y + h + 2), color, 2, cv2.LINE_AA)

def draw_corner_brackets(
    canvas: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    color: tuple[int, int, int],
    thickness: int,
) -> None:

    corner = max(16, int(min(w, h) * 0.22))
    round_r = min(6, corner // 2)
    corners = [
        ((x, y), (1, 0), (0, 1), (180, 270)),
        ((x + w, y), (-1, 0), (0, 1), (270, 360)),
        ((x, y + h), (1, 0), (0, -1), (90, 180)),
        ((x + w, y + h), (-1, 0), (0, -1), (0, 90)),
    ]
    for (cx, cy), (dx, _), (_, dy), (start_angle, end_angle) in corners:
        cv2.line(
            canvas,
            (cx + dx * round_r, cy),
            (cx + dx * corner, cy),
            color,
            thickness,
            cv2.LINE_AA,
        )
        cv2.line(
            canvas,
            (cx, cy + dy * round_r),
            (cx, cy + dy * corner),
            color,
            thickness,
            cv2.LINE_AA,
        )
        cv2.ellipse(
            canvas,
            (cx + dx * round_r, cy + dy * round_r),
            (round_r, round_r),
            0,
            start_angle,
            end_angle,
            color,
            thickness,
            cv2.LINE_AA,
        )

def apply_spotlight(
    canvas: np.ndarray, cx: float, cy: float, w: float, h: float, strength: float = 0.55
) -> np.ndarray:

    canvas_h, canvas_w = canvas.shape[:2]
    small_w, small_h = max(1, canvas_w // 4), max(1, canvas_h // 4)
    sx, sy = small_w / canvas_w, small_h / canvas_h

    mask = np.zeros((small_h, small_w), dtype=np.float32)
    axes = (max(4, int(w * 0.8 * sx)), max(4, int(h * 0.9 * sy)))
    center = (int(cx * sx), int(cy * sy))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(axes) * 0.7)
    mask = np.clip(mask, 0.0, 1.0)
    mask = cv2.resize(mask, (canvas_w, canvas_h), interpolation=cv2.INTER_LINEAR)
    mask3 = mask[:, :, None]

    canvas_f = canvas.astype(np.float32)
    dimmed = canvas_f * (1.0 - strength)
    return (canvas_f * mask3 + dimmed * (1.0 - mask3)).astype(np.uint8)

def draw_status_dot(
    canvas: np.ndarray,
    center: tuple[int, int],
    is_live: bool,
    accent_bright: tuple[int, int, int],
    accent_dim: tuple[int, int, int],
) -> None:

    if not is_live:
        cv2.circle(canvas, center, 5, accent_dim, -1, cv2.LINE_AA)
        return
    pulse = (math.sin(time.monotonic() * 2.4) + 1.0) / 2.0
    glow_r = int(8 + 5 * pulse)
    overlay = canvas.copy()
    cv2.circle(overlay, center, glow_r, accent_bright, -1, cv2.LINE_AA)
    alpha = 0.12 + 0.18 * pulse
    cv2.addWeighted(overlay, alpha, canvas, 1.0 - alpha, 0, dst=canvas)
    cv2.circle(canvas, center, 5, accent_bright, -1, cv2.LINE_AA)
