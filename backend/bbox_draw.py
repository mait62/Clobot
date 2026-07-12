"""Polished rounded bounding-box overlays (cyan palette)."""

from __future__ import annotations

from typing import Any, Sequence

import cv2
import numpy as np

# Cyan / sky palette (BGR)
DETECT_COLORS = [
    (220, 190, 60),
    (210, 160, 40),
    (230, 200, 120),
    (180, 140, 50),
    (200, 170, 90),
]
ANCHOR_COLOR = (230, 210, 50)
LABEL_BG = (18, 28, 36)
LABEL_FG = (255, 255, 255)


def _norm_to_px(bbox: Sequence[float], w: int, h: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    return (
        int(max(0, min(w - 1, x1 * w))),
        int(max(0, min(h - 1, y1 * h))),
        int(max(0, min(w - 1, x2 * w))),
        int(max(0, min(h - 1, y2 * h))),
    )


def _rounded_rect_mask(w: int, h: int, radius: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    if w < 2 or h < 2:
        return mask
    r = max(1, min(radius, w // 2, h // 2))
    cv2.rectangle(mask, (r, 0), (w - r - 1, h - 1), 255, -1)
    cv2.rectangle(mask, (0, r), (w - 1, h - r - 1), 255, -1)
    cv2.circle(mask, (r, r), r, 255, -1, lineType=cv2.LINE_AA)
    cv2.circle(mask, (w - r - 1, r), r, 255, -1, lineType=cv2.LINE_AA)
    cv2.circle(mask, (r, h - r - 1), r, 255, -1, lineType=cv2.LINE_AA)
    cv2.circle(mask, (w - r - 1, h - r - 1), r, 255, -1, lineType=cv2.LINE_AA)
    return mask


def _rounded_rect(
    img: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color: tuple[int, int, int],
    *,
    thickness: int = 2,
    radius: int = 14,
    fill_alpha: float = 0.14,
) -> None:
    x1, x2 = sorted((max(0, x1), min(img.shape[1] - 1, x2)))
    y1, y2 = sorted((max(0, y1), min(img.shape[0] - 1, y2)))
    bw, bh = x2 - x1, y2 - y1
    if bw < 4 or bh < 4:
        return
    radius = max(4, min(radius, bw // 2, bh // 2))
    roi = img[y1:y2, x1:x2]
    mask = _rounded_rect_mask(bw, bh, radius)

    # Soft fill
    overlay = roi.copy()
    overlay[mask > 0] = color
    blended = cv2.addWeighted(overlay, fill_alpha, roi, 1.0 - fill_alpha, 0)
    roi[:] = np.where(mask[..., None] > 0, blended, roi)

    # Rounded stroke
    edges = cv2.Canny(mask, 80, 160)
    if thickness > 1:
        edges = cv2.dilate(edges, np.ones((thickness, thickness), np.uint8), iterations=1)
    roi[edges > 0] = color


def _draw_label(
    img: np.ndarray,
    text: str,
    x: int,
    y: int,
    color: tuple[int, int, int],
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    pad_x, pad_y = 10, 6
    box_h = th + pad_y * 2
    box_w = tw + pad_x * 2 + 4
    top = max(0, y - box_h - 6)
    left = max(0, min(x, img.shape[1] - box_w))

    overlay = img.copy()
    cv2.rectangle(overlay, (left, top), (left + box_w, top + box_h), LABEL_BG, -1)
    cv2.addWeighted(overlay, 0.88, img, 0.12, 0, img)
    cv2.rectangle(img, (left, top), (left + 4, top + box_h), color, -1)
    cv2.putText(
        img,
        text,
        (left + pad_x + 2, top + pad_y + th - 1),
        font,
        scale,
        LABEL_FG,
        thickness,
        cv2.LINE_AA,
    )


def draw_detections(
    bgr: np.ndarray,
    detections: list[dict[str, Any]],
    *,
    selected_index: int | None = None,
    show_confidence: bool = False,
) -> np.ndarray:
    """Overlay rounded detection boxes. Labels show name only by default."""
    out = bgr.copy()
    h, w = out.shape[:2]
    for i, det in enumerate(detections):
        bbox = det.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = _norm_to_px(bbox, w, h)
        color = DETECT_COLORS[i % len(DETECT_COLORS)]
        thick = 3 if selected_index == i else 2
        radius = 16 if selected_index == i else 12
        _rounded_rect(
            out,
            x1,
            y1,
            x2,
            y2,
            color,
            thickness=thick,
            radius=radius,
            fill_alpha=0.18 if selected_index == i else 0.12,
        )
        label = str(det.get("label", "object")).capitalize()
        if show_confidence:
            label = f"{label}  {float(det.get('confidence', 0.0)):.0%}"
        _draw_label(out, label, x1, y1, color)
    return out


def draw_placement_anchor(
    bgr: np.ndarray,
    anchor_box: Sequence[float],
    *,
    title: str = "Place here",
    guidance: dict[str, Any] | None = None,
) -> np.ndarray:
    """Draw a rounded placement slot with soft fill + label."""
    out = bgr.copy()
    h, w = out.shape[:2]
    x1, y1, x2, y2 = _norm_to_px(anchor_box, w, h)
    color = ANCHOR_COLOR
    _rounded_rect(out, x1, y1, x2, y2, color, thickness=3, radius=18, fill_alpha=0.22)

    hint = title
    if guidance:
        arrows = guidance.get("arrows") or []
        if arrows and arrows != ["HOLD"]:
            hint = f"{title}  ·  {' '.join(arrows)}"
        elif arrows == ["HOLD"]:
            hint = f"{title}  ·  aligned"
    _draw_label(out, hint, x1, y1, color)
    return out


def encode_jpeg_bytes(bgr: np.ndarray, quality: int = 85) -> bytes:
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise ValueError("Failed to encode JPEG")
    return buf.tobytes()
