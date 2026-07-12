"""Persistent UI / engine settings for Smart Organizer."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("smart-organizer-settings")

SETTINGS_PATH = Path(__file__).resolve().parent / "organizer_settings.json"

DEFAULTS: dict[str, Any] = {
    "api_base_url": "http://127.0.0.1:5000",
    "use_remote_api": False,
    "yolo_model": "yolo26s.pt",
    "default_conf": 0.35,
    "imgsz": 640,
    "device": "auto",  # auto | cpu | 0
}

KNOWN_MODELS = [
    "yolo26s.pt",
    "yolo26n.pt",
    "yolo11n.pt",
    "yolo11s.pt",
    "yolov8n.pt",
    "yolov8s.pt",
]


def load_settings() -> dict[str, Any]:
    data = dict(DEFAULTS)
    if SETTINGS_PATH.exists():
        try:
            raw = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                data.update({k: raw[k] for k in DEFAULTS if k in raw})
        except Exception as exc:
            logger.warning("Failed to load settings: %s", exc)
    return data


def save_settings(updates: dict[str, Any]) -> dict[str, Any]:
    data = load_settings()
    for key, value in updates.items():
        if key in DEFAULTS:
            data[key] = value
    SETTINGS_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data


def apply_to_vision() -> dict[str, Any]:
    """Push settings into the local vision engine (app.py)."""
    import app as vision

    data = load_settings()
    vision.DEFAULT_PRED_CONF = float(data["default_conf"])
    vision.DEFAULT_IMG_SIZE = int(data["imgsz"])
    device = str(data.get("device") or "auto")
    if device == "cpu":
        vision.INFER_DEVICE = "cpu"
    elif device == "0":
        vision.INFER_DEVICE = 0
    # auto leaves current / resolve on next model load

    model = str(data.get("yolo_model") or DEFAULTS["yolo_model"]).strip()
    vision.reload_detect_model(model, device=device)
    return data
