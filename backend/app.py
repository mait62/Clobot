"""
Smart Organizer Robot — Flask API for the Next.js frontend.

Routes used by the app:
  GET  /api/health
  POST /api/detect-object
  POST /api/crop-and-identify
  POST /api/suggest-spatial-slot
  POST /api/calculate-placement
  POST /api/pipeline
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import time
import traceback
import uuid
from typing import Any

import cv2
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("smart-organizer")

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

DETECT_WEIGHTS = os.environ.get("YOLO_MODEL", "yolo26s.pt").strip() or "yolo26s.pt"
SEG_WEIGHTS = "yolo11n-seg.pt"
DEFAULT_PRED_CONF = float(os.environ.get("YOLO_CONF", "0.15"))
DEFAULT_IMG_SIZE = int(os.environ.get("YOLO_IMGSZ", "640"))

detect_model: YOLO | None = None
seg_model: YOLO | None = None
seg_available = False
INFER_DEVICE: str | int = "cpu"

SESSION_MEMORY: dict[str, dict[str, Any]] = {}
SESSION_TTL_SEC = 60 * 30

PLACEMENT_MAP: dict[str, dict[str, Any]] = {
    "bottle": {"target": "Shelf A - Slot 1", "coordinates": {"x": -1.8, "y": 1.4, "z": -0.6}, "rotation": 0},
    "cup": {"target": "Shelf A - Slot 3", "coordinates": {"x": -0.6, "y": 1.4, "z": -0.6}, "rotation": 15},
    "wine glass": {"target": "Shelf A - Slot 2", "coordinates": {"x": -1.2, "y": 1.4, "z": -0.6}, "rotation": 0},
    "book": {"target": "Shelf B - Slot 4", "coordinates": {"x": 2.1, "y": 1.2, "z": -1.5}, "rotation": 90},
    "cell phone": {"target": "Shelf C - Slot 1", "coordinates": {"x": 1.5, "y": 0.6, "z": -0.8}, "rotation": 45},
    "remote": {"target": "Shelf C - Slot 2", "coordinates": {"x": 2.0, "y": 0.6, "z": -0.8}, "rotation": 0},
    "laptop": {"target": "Shelf B - Slot 1", "coordinates": {"x": 0.8, "y": 1.2, "z": -1.5}, "rotation": 0},
    "keyboard": {"target": "Shelf B - Slot 2", "coordinates": {"x": 1.4, "y": 1.2, "z": -1.5}, "rotation": 0},
    "mouse": {"target": "Shelf C - Slot 3", "coordinates": {"x": 2.4, "y": 0.6, "z": -0.8}, "rotation": 30},
    "scissors": {"target": "Drawer D - Slot 1", "coordinates": {"x": -2.0, "y": 0.3, "z": -1.2}, "rotation": 90},
    "knife": {"target": "Drawer D - Slot 2", "coordinates": {"x": -1.5, "y": 0.3, "z": -1.2}, "rotation": 90},
    "backpack": {"target": "Shelf E - Slot 1", "coordinates": {"x": -2.2, "y": 0.9, "z": -2.0}, "rotation": 0},
    "handbag": {"target": "Shelf E - Slot 2", "coordinates": {"x": -1.4, "y": 0.9, "z": -2.0}, "rotation": 0},
    "umbrella": {"target": "Entry Stand", "coordinates": {"x": 0.0, "y": 1.0, "z": -2.2}, "rotation": 0},
    "clock": {"target": "Shelf B - Slot 5", "coordinates": {"x": 2.6, "y": 1.8, "z": -1.5}, "rotation": 0},
    "teddy bear": {"target": "Shelf E - Slot 3", "coordinates": {"x": -0.6, "y": 0.9, "z": -2.0}, "rotation": 20},
    "sports ball": {"target": "Bin F", "coordinates": {"x": 2.5, "y": 0.2, "z": -2.0}, "rotation": 0},
    "apple": {"target": "Shelf A - Slot 4", "coordinates": {"x": 0.0, "y": 1.4, "z": -0.6}, "rotation": 0},
    "banana": {"target": "Shelf A - Slot 5", "coordinates": {"x": 0.5, "y": 1.4, "z": -0.6}, "rotation": 0},
    "orange": {"target": "Shelf A - Slot 4", "coordinates": {"x": 0.0, "y": 1.4, "z": -0.6}, "rotation": 0},
}

DEFAULT_PLACEMENT = {
    "target": "General Storage - Bin G",
    "coordinates": {"x": 0.0, "y": 0.5, "z": -1.8},
    "rotation": 0,
}

GRID_ROWS = 3
GRID_COLS = 4
PREFERRED_SLOT_ORDER = [
    (1, 1), (1, 2), (0, 1), (0, 2), (1, 0), (1, 3),
    (2, 1), (2, 2), (0, 0), (0, 3), (2, 0), (2, 3),
]


def _resolve_device() -> str | int:
    try:
        import torch

        if torch.cuda.is_available():
            logger.info("CUDA available — GPU 0 (%s)", torch.cuda.get_device_name(0))
            return 0
        logger.info("CUDA not available — using CPU")
    except Exception as exc:
        logger.warning("CUDA probe failed (%s) — using CPU", exc)
    return "cpu"


def init_models() -> YOLO:
    global detect_model, seg_model, seg_available, INFER_DEVICE
    if detect_model is None:
        INFER_DEVICE = _resolve_device()
        logger.info("Loading %s on device=%s", DETECT_WEIGHTS, INFER_DEVICE)
        detect_model = YOLO(DETECT_WEIGHTS)
        logger.info("Detect model ready.")
    if seg_model is None:
        try:
            seg_model = YOLO(SEG_WEIGHTS)
            seg_available = True
            logger.info("Seg model ready: %s", SEG_WEIGHTS)
        except Exception as exc:
            seg_available = False
            logger.warning("Seg model unavailable: %s", exc)
    return detect_model


def reload_detect_model(weights: str, device: str = "auto") -> dict[str, Any]:
    """Hot-swap the detection weights (used by /backend settings)."""
    global detect_model, DETECT_WEIGHTS, INFER_DEVICE
    weights = (weights or "").strip() or DETECT_WEIGHTS
    if device == "cpu":
        INFER_DEVICE = "cpu"
    elif device == "0":
        INFER_DEVICE = 0
    else:
        INFER_DEVICE = _resolve_device()

    logger.info("Reloading detect model %s on device=%s", weights, INFER_DEVICE)
    detect_model = YOLO(weights)
    DETECT_WEIGHTS = weights
    logger.info("Detect model reloaded: %s", weights)
    # Ensure seg is initialized too
    init_models()
    return {
        "model": DETECT_WEIGHTS,
        "device": INFER_DEVICE,
        "seg_available": seg_available,
    }


def model_status() -> dict[str, Any]:
    return {
        "model": DETECT_WEIGHTS,
        "model_loaded": detect_model is not None,
        "device": INFER_DEVICE if detect_model is not None else _resolve_device(),
        "seg_available": seg_available,
        "default_conf": DEFAULT_PRED_CONF,
        "imgsz": DEFAULT_IMG_SIZE,
    }


def _error(message: str, status: int = 400, detail: str | None = None):
    body: dict[str, Any] = {"error": message, "success": False}
    if detail:
        body["detail"] = detail
    return jsonify(body), status


def _load_image_from_request(field: str = "image") -> Image.Image:
    if field in request.files:
        file = request.files[field]
        if not file or file.filename == "":
            raise ValueError("Empty image file uploaded.")
        data = file.read()
        if not data:
            raise ValueError("Uploaded image payload is empty.")
        return Image.open(io.BytesIO(data)).convert("RGB")
    if "image" in request.files and field != "image":
        return _load_image_from_request("image")
    if request.data and field == "image":
        return Image.open(io.BytesIO(request.data)).convert("RGB")
    raise ValueError("No image provided. Send multipart field 'image'.")


def _pil_to_bgr(image: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def _purge_expired_sessions() -> None:
    now = time.time()
    for sid in [
        s for s, t in SESSION_MEMORY.items() if now - t.get("created_at", 0) > SESSION_TTL_SEC
    ]:
        SESSION_MEMORY.pop(sid, None)


def _run_detections(
    image: Image.Image,
    *,
    conf: float | None = None,
    imgsz: int | None = None,
) -> list[dict[str, Any]]:
    yolo = init_models()
    results = yolo.predict(
        image,
        device=INFER_DEVICE,
        conf=DEFAULT_PRED_CONF if conf is None else conf,
        imgsz=DEFAULT_IMG_SIZE if imgsz is None else imgsz,
        verbose=False,
    )
    result = results[0]
    detections: list[dict[str, Any]] = []
    if result.boxes is None or len(result.boxes) == 0:
        return detections
    for box in result.boxes:
        cls_id = int(box.cls[0])
        score = float(box.conf[0])
        detections.append(
            {
                "label": yolo.names[cls_id],
                "confidence": round(score, 4),
                "bbox": [round(v, 6) for v in box.xyxyn[0].tolist()],
                "bbox_pixels": [float(v) for v in box.xyxy[0].tolist()],
            }
        )
    detections.sort(key=lambda d: d["confidence"], reverse=True)
    return detections


def _point_in_bbox(nx: float, ny: float, bbox: list[float]) -> bool:
    x1, y1, x2, y2 = bbox
    return x1 <= nx <= x2 and y1 <= ny <= y2


def _select_detection_by_point(
    detections: list[dict[str, Any]], click_x: float, click_y: float
) -> dict[str, Any] | None:
    containing = [d for d in detections if _point_in_bbox(click_x, click_y, d["bbox"])]
    if not containing:
        return None

    def area(d: dict[str, Any]) -> float:
        x1, y1, x2, y2 = d["bbox"]
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    return min(containing, key=area)


def _clamp_bbox(bbox: list[float]) -> list[float]:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    x1, x2 = sorted((max(0.0, min(1.0, x1)), max(0.0, min(1.0, x2))))
    y1, y2 = sorted((max(0.0, min(1.0, y1)), max(0.0, min(1.0, y2))))
    if x2 - x1 < 0.02:
        cx = (x1 + x2) / 2
        x1, x2 = max(0.0, cx - 0.05), min(1.0, cx + 0.05)
    if y2 - y1 < 0.02:
        cy = (y1 + y2) / 2
        y1, y2 = max(0.0, cy - 0.05), min(1.0, cy + 0.05)
    return [round(x1, 6), round(y1, 6), round(x2, 6), round(y2, 6)]


def _crop_bgr(bgr: np.ndarray, bbox_norm: list[float]) -> np.ndarray:
    h, w = bgr.shape[:2]
    x1, y1, x2, y2 = bbox_norm
    px1, py1 = max(0, int(x1 * w)), max(0, int(y1 * h))
    px2, py2 = min(w, int(x2 * w)), min(h, int(y2 * h))
    if px2 <= px1 or py2 <= py1:
        raise ValueError("Invalid crop region.")
    return bgr[py1:py2, px1:px2].copy()


def _dominant_color_hex(crop_bgr: np.ndarray) -> str:
    if crop_bgr.size == 0:
        return "#64748b"
    small = cv2.resize(crop_bgr, (48, 48), interpolation=cv2.INTER_AREA)
    pixels = small.reshape(-1, 3).astype(np.float32)
    mask = pixels.mean(axis=1) > 25
    sample = pixels[mask] if mask.any() else pixels
    b, g, r = sample.mean(axis=0)
    return "#{:02x}{:02x}{:02x}".format(int(r), int(g), int(b))


def _approx_dimensions_cm(bbox_norm: list[float], image_w: int, image_h: int) -> dict[str, float]:
    x1, y1, x2, y2 = bbox_norm
    fov_w_cm = 50.0
    fov_h_cm = fov_w_cm * (image_h / max(image_w, 1))
    width_cm = round((x2 - x1) * fov_w_cm, 1)
    height_cm = round((y2 - y1) * fov_h_cm, 1)
    return {
        "width_cm": width_cm,
        "height_cm": height_cm,
        "aspect": round(width_cm / max(height_cm, 0.1), 2),
    }


def _encode_crop_jpeg_b64(crop_bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", crop_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        raise ValueError("Failed to encode crop preview.")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _try_refine_bbox_with_seg(image: Image.Image, bbox_norm: list[float]) -> list[float]:
    if not seg_available or seg_model is None:
        return bbox_norm
    try:
        results = seg_model.predict(image, device=INFER_DEVICE, verbose=False)
        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return bbox_norm
        cx = (bbox_norm[0] + bbox_norm[2]) / 2
        cy = (bbox_norm[1] + bbox_norm[3]) / 2
        best, best_area = None, 1e9
        for box in result.boxes:
            xyxyn = box.xyxyn[0].tolist()
            if _point_in_bbox(cx, cy, xyxyn):
                area = (xyxyn[2] - xyxyn[0]) * (xyxyn[3] - xyxyn[1])
                if area < best_area:
                    best_area, best = area, xyxyn
        return _clamp_bbox(best) if best else bbox_norm
    except Exception:
        return bbox_norm


def _slot_rect(row: int, col: int, inset: float = 0.04) -> list[float]:
    cell_w, cell_h = 1.0 / GRID_COLS, 1.0 / GRID_ROWS
    return [
        round(col * cell_w + inset * cell_w, 4),
        round(row * cell_h + inset * cell_h, 4),
        round((col + 1) * cell_w - inset * cell_w, 4),
        round((row + 1) * cell_h - inset * cell_h, 4),
    ]


def _bbox_center(bbox: list[float]) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)


def _cell_occupied(
    row: int, col: int, occupied_bboxes: list[list[float]], overlap_thresh: float = 0.18
) -> bool:
    sx1, sy1, sx2, sy2 = _slot_rect(row, col, inset=0.0)
    sw, sh = sx2 - sx1, sy2 - sy1
    for ox1, oy1, ox2, oy2 in occupied_bboxes:
        ix1, iy1 = max(sx1, ox1), max(sy1, oy1)
        ix2, iy2 = min(sx2, ox2), min(sy2, oy2)
        if ix2 <= ix1 or iy2 <= iy1:
            continue
        if ((ix2 - ix1) * (iy2 - iy1)) / max(sw * sh, 1e-6) >= overlap_thresh:
            return True
    return False


def _guidance_arrows(anchor: list[float]) -> dict[str, Any]:
    ax, ay = _bbox_center(anchor)
    dx, dy = ax - 0.5, ay - 0.5
    arrows: list[str] = []
    if abs(dx) > 0.08:
        arrows.append("RIGHT" if dx > 0 else "LEFT")
    if abs(dy) > 0.08:
        arrows.append("DOWN" if dy > 0 else "UP")
    if not arrows:
        arrows.append("HOLD")
    return {
        "arrows": arrows,
        "dx": round(dx, 4),
        "dy": round(dy, 4),
        "magnitude": round(min(1.0, (dx * dx + dy * dy) ** 0.5 * 2.2), 3),
        "aligned": arrows == ["HOLD"],
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "ok",
            "model": DETECT_WEIGHTS,
            "seg_available": seg_available,
            "model_loaded": detect_model is not None,
            "device": INFER_DEVICE if detect_model is not None else _resolve_device(),
            "service": "Smart Organizer Robot",
        }
    )


@app.route("/api/detect-object", methods=["POST"])
def detect_object():
    try:
        image = _load_image_from_request()
    except ValueError as exc:
        return _error(str(exc), 400)
    except Exception as exc:
        return _error("Failed to decode image.", 400, str(exc))

    try:
        min_conf = DEFAULT_PRED_CONF
        raw_min = request.form.get("min_confidence") or request.args.get("min_confidence")
        if raw_min is not None:
            min_conf = max(0.05, min(0.95, float(raw_min)))

        detections = [
            d for d in _run_detections(image, conf=min_conf) if d["confidence"] >= min_conf
        ]
        top = detections[0] if detections else None
        w, h = image.size
        return jsonify(
            {
                "success": True,
                "label": top["label"] if top else None,
                "confidence": top["confidence"] if top else 0.0,
                "bbox": top["bbox"] if top else None,
                "detections": detections,
                "count": len(detections),
                "image_size": {"width": w, "height": h},
                "model": DETECT_WEIGHTS,
                "device": INFER_DEVICE,
            }
        )
    except Exception as exc:
        logger.error("Detection failed:\n%s", traceback.format_exc())
        return _error("Inference failed.", 500, str(exc))


@app.route("/api/crop-and-identify", methods=["POST"])
def crop_and_identify():
    try:
        image = _load_image_from_request()
    except ValueError as exc:
        return _error(str(exc), 400)
    except Exception as exc:
        return _error("Failed to decode image.", 400, str(exc))

    form = request.form
    click_x, click_y = form.get("click_x"), form.get("click_y")
    bbox_raw = form.get("bbox") or form.get("bbox[]")

    user_bbox: list[float] | None = None
    if bbox_raw:
        try:
            cleaned = bbox_raw.strip()
            user_bbox = (
                [float(v) for v in json.loads(cleaned)]
                if cleaned.startswith("[")
                else [float(v) for v in cleaned.split(",")]
            )
            if len(user_bbox) != 4:
                return _error("bbox must be [x1, y1, x2, y2] normalized.")
            user_bbox = _clamp_bbox(user_bbox)
        except Exception:
            return _error("Invalid bbox payload.")

    click: tuple[float, float] | None = None
    if click_x is not None and click_y is not None:
        try:
            click = (float(click_x), float(click_y))
            if not (0.0 <= click[0] <= 1.0 and 0.0 <= click[1] <= 1.0):
                return _error("click_x / click_y must be normalized to [0, 1].")
        except ValueError:
            return _error("click_x / click_y must be numeric.")

    if user_bbox is None and click is None:
        return _error("Provide click_x+click_y or bbox with the image.")

    try:
        min_conf = DEFAULT_PRED_CONF
        raw_min = form.get("min_confidence")
        if raw_min is not None:
            try:
                min_conf = max(0.05, min(0.95, float(raw_min)))
            except ValueError:
                pass

        detections = [
            d for d in _run_detections(image, conf=min_conf) if d["confidence"] >= min_conf
        ]
        selected: dict[str, Any] | None = None
        selection_mode = "bbox"

        if user_bbox is not None:
            cx = (user_bbox[0] + user_bbox[2]) / 2
            cy = (user_bbox[1] + user_bbox[3]) / 2
            selected = _select_detection_by_point(detections, cx, cy)
            if selected:
                d = selected["bbox"]
                crop_bbox = _clamp_bbox(
                    [
                        max(user_bbox[0], d[0]),
                        max(user_bbox[1], d[1]),
                        min(user_bbox[2], d[2]),
                        min(user_bbox[3], d[3]),
                    ]
                )
                selection_mode = "bbox+detect"
            else:
                crop_bbox = user_bbox
                selected = {"label": "object", "confidence": 0.5, "bbox": user_bbox}
                selection_mode = "bbox"
        else:
            assert click is not None
            selected = _select_detection_by_point(detections, click[0], click[1])
            if selected:
                crop_bbox = selected["bbox"]
                selection_mode = "point+detect"
            else:
                pad = 0.12
                crop_bbox = _clamp_bbox(
                    [click[0] - pad, click[1] - pad, click[0] + pad, click[1] + pad]
                )
                selected = {"label": "object", "confidence": 0.35, "bbox": crop_bbox}
                selection_mode = "point-fallback"

        crop_bbox = _try_refine_bbox_with_seg(image, crop_bbox)
        bgr = _pil_to_bgr(image)
        crop = _crop_bgr(bgr, crop_bbox)
        color_hex = _dominant_color_hex(crop)
        dims = _approx_dimensions_cm(crop_bbox, *image.size)
        crop_b64 = _encode_crop_jpeg_b64(crop)

        crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        crop_dets = _run_detections(crop_pil)
        if crop_dets and crop_dets[0]["confidence"] >= 0.35:
            label, confidence = crop_dets[0]["label"], crop_dets[0]["confidence"]
        else:
            label, confidence = selected["label"], float(selected["confidence"])

        _purge_expired_sessions()
        session_id = request.form.get("session_id") or str(uuid.uuid4())
        token = {
            "session_id": session_id,
            "created_at": time.time(),
            "label": label,
            "confidence": round(confidence, 4),
            "bbox": crop_bbox,
            "click": {"x": click[0], "y": click[1]} if click else None,
            "color_hex": color_hex,
            "dimensions": dims,
            "selection_mode": selection_mode,
            "crop_jpeg_b64": crop_b64,
            "placement_hint": PLACEMENT_MAP.get(label.lower(), {**DEFAULT_PLACEMENT}),
        }
        SESSION_MEMORY[session_id] = token

        return jsonify(
            {
                "success": True,
                "session_id": session_id,
                "label": label,
                "confidence": round(confidence, 4),
                "bbox": crop_bbox,
                "color_hex": color_hex,
                "dimensions": dims,
                "selection_mode": selection_mode,
                "crop_preview_b64": crop_b64,
                "detections": detections[:12],
                "token": {
                    "label": label,
                    "color_hex": color_hex,
                    "dimensions": dims,
                    "placement_hint": token["placement_hint"],
                },
                "seg_used": seg_available,
                "model": DETECT_WEIGHTS,
            }
        )
    except Exception as exc:
        logger.error("crop-and-identify failed:\n%s", traceback.format_exc())
        return _error("Crop/identify failed.", 500, str(exc))


@app.route("/api/suggest-spatial-slot", methods=["POST"])
def suggest_spatial_slot():
    try:
        image = _load_image_from_request()
    except ValueError as exc:
        return _error(str(exc), 400)
    except Exception as exc:
        return _error("Failed to decode image.", 400, str(exc))

    session_id = request.form.get("session_id") or (request.get_json(silent=True) or {}).get(
        "session_id"
    )
    _purge_expired_sessions()
    token = SESSION_MEMORY.get(session_id) if session_id else None
    label_override = request.form.get("label")

    if token is None:
        if label_override:
            label = label_override.strip().lower()
            token = {
                "label": label,
                "placement_hint": PLACEMENT_MAP.get(label, {**DEFAULT_PLACEMENT}),
                "color_hex": "#2563eb",
                "dimensions": {"width_cm": 10, "height_cm": 10, "aspect": 1.0},
            }
        else:
            return _error(
                "Unknown or expired session_id. Run /api/crop-and-identify first.",
                404,
            )

    try:
        from placement_logic import suggest_similar_placement

        detections = _run_detections(image)
        item_label = str(token.get("label") or label_override or "object")
        place = suggest_similar_placement(item_label, detections, min_conf=0.25)

        return jsonify(
            {
                "success": True,
                "session_id": session_id,
                "label": item_label,
                "target": place["target"],
                "slot": {"mode": place["mode"], "similar_count": place["similar_count"]},
                "anchor_box": place["anchor_box"],
                "guidance": place["guidance"],
                "recommendation": place["recommendation"],
                "coordinates": place["coordinates"],
                "rotation": place["rotation"],
                "object_color": token.get("color_hex"),
                "detections_in_view": len(detections),
                "model": DETECT_WEIGHTS,
            }
        )
    except Exception as exc:
        logger.error("suggest-spatial-slot failed:\n%s", traceback.format_exc())
        return _error("Spatial suggestion failed.", 500, str(exc))


@app.route("/api/calculate-placement", methods=["POST"])
def calculate_placement():
    label: str | None = None
    storage_scanned = False

    if request.is_json:
        payload = request.get_json(silent=True) or {}
        label = payload.get("label") or payload.get("object") or payload.get("class")
    elif request.form:
        label = request.form.get("label") or request.form.get("object")

    if "image" in request.files or "storage" in request.files:
        storage_scanned = True

    if not label or not str(label).strip():
        return _error('Missing object label. Provide {"label": "cup"} or form field label.')

    key = str(label).strip().lower()
    placement = PLACEMENT_MAP.get(key, {**DEFAULT_PLACEMENT})
    return jsonify(
        {
            "success": True,
            "label": key,
            "known_class": key in PLACEMENT_MAP,
            "storage_scanned": storage_scanned,
            "target": placement["target"],
            "coordinates": placement["coordinates"],
            "rotation": placement["rotation"],
            "inventory_matrix": {
                "shelf": placement["target"].split(" - ")[0]
                if " - " in placement["target"]
                else placement["target"],
                "slot": placement["target"],
            },
        }
    )


@app.route("/api/pipeline", methods=["POST"])
def pipeline():
    try:
        image = _load_image_from_request()
    except ValueError as exc:
        return _error(str(exc), 400)
    except Exception as exc:
        return _error("Failed to decode image.", 400, str(exc))

    try:
        detections = _run_detections(image)
        top = detections[0] if detections else None
        if not top:
            return jsonify(
                {
                    "success": True,
                    "label": None,
                    "confidence": 0.0,
                    "bbox": None,
                    "detections": [],
                    "placement": None,
                    "message": "No objects detected.",
                }
            )

        key = top["label"].lower()
        placement = PLACEMENT_MAP.get(key, {**DEFAULT_PLACEMENT})
        return jsonify(
            {
                "success": True,
                "label": top["label"],
                "confidence": top["confidence"],
                "bbox": top["bbox"],
                "detections": detections,
                "placement": {
                    "target": placement["target"],
                    "coordinates": placement["coordinates"],
                    "rotation": placement["rotation"],
                    "known_class": key in PLACEMENT_MAP,
                },
            }
        )
    except Exception as exc:
        logger.error("Pipeline failed:\n%s", traceback.format_exc())
        return _error("Pipeline failed.", 500, str(exc))


if __name__ == "__main__":
    init_models()
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)
