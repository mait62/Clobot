"""Optional remote Flask API client for the NiceGUI app."""

from __future__ import annotations

import io
import logging
from typing import Any

import requests
from PIL import Image

logger = logging.getLogger("smart-organizer-api")


def _base(url: str) -> str:
    return (url or "").rstrip("/")


def health_check(api_base_url: str, timeout: float = 4.0) -> dict[str, Any]:
    base = _base(api_base_url)
    try:
        res = requests.get(f"{base}/api/health", timeout=timeout)
        data = res.json() if res.content else {}
        return {
            "ok": res.ok,
            "status_code": res.status_code,
            "data": data,
            "error": None if res.ok else data.get("error") or res.text[:200],
        }
    except Exception as exc:
        return {"ok": False, "status_code": 0, "data": {}, "error": str(exc)}


def detect_remote(
    api_base_url: str,
    image: Image.Image,
    min_confidence: float,
    timeout: float = 60.0,
) -> list[dict[str, Any]]:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=90)
    buf.seek(0)
    base = _base(api_base_url)
    res = requests.post(
        f"{base}/api/detect-object",
        files={"image": ("frame.jpg", buf, "image/jpeg")},
        data={"min_confidence": str(min_confidence)},
        timeout=timeout,
    )
    data = res.json()
    if not res.ok or data.get("error"):
        raise RuntimeError(data.get("error") or f"Remote detect failed ({res.status_code})")
    return list(data.get("detections") or [])
