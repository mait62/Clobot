"""
Smart Organizer — NiceGUI frontend.

Multi-step: Photo → Select item → Find spot → Done.
Browser-native camera (no JPEG flicker). Click / drag to select.
Optional custom label. Placement groups similar items — no fake bin names.

Run (from backend/):
  python ui_app.py
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import time
import uuid
from typing import Any

import cv2
import numpy as np
from nicegui import app as nicegui_app
from nicegui import ui
from PIL import Image

import app as vision
from bbox_draw import draw_detections, draw_placement_anchor, encode_jpeg_bytes
from placement_logic import suggest_similar_placement

from ui_chrome import inject_styles, render_header
import settings as cfg
import api_client
import backend_page  # noqa: F401  — registers /backend route


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("smart-organizer-ui")

STEPS = ("photo", "pick", "place", "done")
STEP_LABELS = {
    "photo": "Photo",
    "pick": "Select item",
    "place": "Find spot",
    "done": "Done",
}


def bgr_to_data_url(bgr: np.ndarray, quality: int = 85) -> str:
    raw = encode_jpeg_bytes(bgr, quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(raw).decode("ascii")


def pil_from_bytes(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert("RGB")


def pil_to_bgr(image: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def data_url_to_bgr(data_url: str) -> np.ndarray:
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    raw = base64.b64decode(data_url)
    return pil_to_bgr(pil_from_bytes(raw))


def read_upload_bytes(e: Any) -> bytes:
    content = e.content.read() if hasattr(e.content, "read") else e.content
    if isinstance(content, memoryview):
        return content.tobytes()
    if isinstance(content, bytes):
        return content
    return bytes(content)


def create_session(
    label: str,
    confidence: float,
    crop_bbox: list[float],
    crop_bgr: np.ndarray,
    pil_size: tuple[int, int],
) -> tuple[str, str]:
    color_hex = vision._dominant_color_hex(crop_bgr)
    dims = vision._approx_dimensions_cm(crop_bbox, *pil_size)
    crop_b64 = vision._encode_crop_jpeg_b64(crop_bgr)
    session_id = str(uuid.uuid4())
    vision.SESSION_MEMORY[session_id] = {
        "session_id": session_id,
        "created_at": time.time(),
        "label": label,
        "confidence": round(confidence, 4),
        "bbox": crop_bbox,
        "color_hex": color_hex,
        "dimensions": dims,
        "crop_jpeg_b64": crop_b64,
    }
    return session_id, crop_b64


class OrganizerState:
    def __init__(self) -> None:
        self.step = "photo"
        self.detection_strength = 0.35
        self.live_running = False
        self.placement_overlay = False
        self.session_id: str | None = None
        self.selected_label: str | None = None
        self.crop_preview_b64: str | None = None
        self.last_detections: list[dict[str, Any]] = []
        self.placement: dict[str, Any] | None = None
        self.object_bgr: np.ndarray | None = None
        self.place_bgr: np.ndarray | None = None
        self.freeze_url: str | None = None
        self.status = "Take a photo or upload one to begin"
        self.error: str | None = None
        self._infer_busy = False
        # Selection draft (normalized)
        self.drag_start: tuple[float, float] | None = None
        self.draft_bbox: list[float] | None = None
        self.img_w = 1
        self.img_h = 1



# Browser camera helpers — no server frame pumping
CAM_JS = """
window.__soCam = {
  stream: null,
  async list() {
    try {
      // permission nudge so labels appear
      const tmp = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      tmp.getTracks().forEach(t => t.stop());
    } catch (_) {}
    const all = await navigator.mediaDevices.enumerateDevices();
    return all.filter(d => d.kind === 'videoinput').map((d, i) => ({
      id: d.deviceId, label: d.label || ('Camera ' + (i + 1))
    }));
  },
  async start(deviceId) {
    this.stop();
    const constraints = deviceId
      ? { video: { deviceId: { exact: deviceId } }, audio: false }
      : { video: { facingMode: { ideal: 'environment' } }, audio: false };
    let stream;
    try { stream = await navigator.mediaDevices.getUserMedia(constraints); }
    catch (_) { stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false }); }
    this.stream = stream;
    const v = document.getElementById('so-live-video');
    if (v) { v.srcObject = stream; await v.play().catch(() => {}); }
    return true;
  },
  stop() {
    if (this.stream) { this.stream.getTracks().forEach(t => t.stop()); this.stream = null; }
    const v = document.getElementById('so-live-video');
    if (v) v.srcObject = null;
  },
  snapshot() {
    const v = document.getElementById('so-live-video');
    if (!v || !v.videoWidth) return null;
    const c = document.createElement('canvas');
    c.width = v.videoWidth; c.height = v.videoHeight;
    c.getContext('2d').drawImage(v, 0, 0);
    return c.toDataURL('image/jpeg', 0.92);
  },
  contentLayout() {
    const v = document.getElementById('so-live-video');
    const layer = document.getElementById('so-overlay-layer');
    if (!v || !layer || !v.videoWidth) return null;
    const vr = v.getBoundingClientRect();
    const lr = layer.getBoundingClientRect();
    if (vr.width < 2 || vr.height < 2 || lr.width < 2 || lr.height < 2) return null;
    const videoAspect = v.videoWidth / v.videoHeight;
    const elemAspect = vr.width / vr.height;
    let w, h, ox, oy;
    if (videoAspect > elemAspect) {
      w = vr.width; h = vr.width / videoAspect;
      ox = 0; oy = (vr.height - h) / 2;
    } else {
      h = vr.height; w = vr.height * videoAspect;
      oy = 0; ox = (vr.width - w) / 2;
    }
    return {
      left: ((vr.left - lr.left + ox) / lr.width) * 100,
      top: ((vr.top - lr.top + oy) / lr.height) * 100,
      width: (w / lr.width) * 100,
      height: (h / lr.height) * 100,
    };
  },
  drawOverlays(payload) {
    const layer = document.getElementById('so-overlay-layer');
    if (!layer) return false;
    const layout = this.contentLayout();
    layer.innerHTML = '';
    if (!layout || !payload) return false;
    const boxes = payload.boxes || [];
    const colors = ['#22d3ee', '#38bdf8', '#67e8f9', '#2dd4bf', '#818cf8'];
    boxes.forEach((b, i) => {
      const el = document.createElement('div');
      el.className = 'so-live-box' + (b.anchor ? ' is-anchor' : '');
      const color = b.anchor ? '#fbbf24' : colors[i % colors.length];
      const x = layout.left + b.x1 * layout.width;
      const y = layout.top + b.y1 * layout.height;
      const w = (b.x2 - b.x1) * layout.width;
      const h = (b.y2 - b.y1) * layout.height;
      el.style.left = x + '%';
      el.style.top = y + '%';
      el.style.width = w + '%';
      el.style.height = h + '%';
      el.style.borderColor = color;
      el.style.background = b.anchor ? 'rgba(251,191,36,0.18)' : 'rgba(34,211,238,0.12)';
      if (b.label) {
        const tag = document.createElement('span');
        tag.className = 'so-live-tag';
        tag.textContent = b.label;
        tag.style.borderLeftColor = color;
        el.appendChild(tag);
      }
      layer.appendChild(el);
    });
    return true;
  },
  clearOverlays() {
    const layer = document.getElementById('so-overlay-layer');
    if (layer) layer.innerHTML = '';
  }
};
"""



def run_detections_for_ui(pil_image, conf: float):
    """Local YOLO or remote Flask /api/detect-object based on settings."""
    conf = float(conf)
    opts = cfg.load_settings()
    if opts.get("use_remote_api"):
        return api_client.detect_remote(
            str(opts.get("api_base_url") or cfg.DEFAULTS["api_base_url"]),
            pil_image,
            conf,
        )
    return [
        d
        for d in vision._run_detections(pil_image, conf=conf)
        if d["confidence"] >= conf
    ]


@ui.page("/")
def main_page() -> None:
    state = OrganizerState()
    inject_styles()
    ui.add_body_html(f"<script>{CAM_JS}</script>")
    try:
        cfg.apply_to_vision()
    except Exception:
        vision.init_models()
    saved = cfg.load_settings()
    state.detection_strength = float(saved.get("default_conf", 0.35))

    camera_options: dict[str, str] = {"": "Default camera"}
    place_timer: Any = None

    # refs
    live_video: ui.element
    freeze_img: ui.interactive_image
    frame_wrap: ui.element
    status_label: ui.label
    error_label: ui.label
    step_row: ui.element
    cam_btn: ui.button
    place_cam_btn: ui.button
    cam_select: ui.select
    step_panels: dict[str, ui.element] = {}
    pick_list: ui.column
    result_card: ui.column
    place_toggle: ui.switch
    place_summary: ui.column
    label_input: ui.input
    overlay_layer: ui.element
    page_root: ui.element

    def set_status(msg: str) -> None:
        state.status = msg
        status_label.set_text(msg)

    def set_error(msg: str | None) -> None:
        state.error = msg
        error_label.set_text(msg or "")
        error_label.set_visibility(bool(msg))

    def show_live(on: bool) -> None:
        live_video.set_visibility(on)
        freeze_img.set_visibility(not on and bool(state.freeze_url))
        if on:
            frame_wrap.classes(remove="is-empty")
        elif not state.freeze_url:
            frame_wrap.classes(add="is-empty")

    def show_freeze(url: str) -> None:
        state.freeze_url = url
        freeze_img.set_source(url)
        live_video.set_visibility(False)
        freeze_img.set_visibility(True)
        frame_wrap.classes(remove="is-empty")
        clear_live_overlays()

    def clear_preview() -> None:
        state.freeze_url = None
        freeze_img.set_source("")
        freeze_img.set_visibility(False)
        live_video.set_visibility(False)
        frame_wrap.classes(add="is-empty")
        clear_live_overlays()

    def clear_live_overlays() -> None:
        try:
            ui.run_javascript("window.__soCam && window.__soCam.clearOverlays()")
        except Exception:
            pass

    async def draw_live_overlays(
        *,
        detections: list[dict[str, Any]] | None = None,
        anchor: list[float] | None = None,
        anchor_label: str | None = None,
    ) -> None:
        boxes: list[dict[str, Any]] = []
        for det in detections or []:
            bbox = det.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            boxes.append(
                {
                    "x1": float(bbox[0]),
                    "y1": float(bbox[1]),
                    "x2": float(bbox[2]),
                    "y2": float(bbox[3]),
                    "label": str(det.get("label", "object")).capitalize(),
                    "anchor": False,
                }
            )
        if anchor and len(anchor) == 4:
            boxes.append(
                {
                    "x1": float(anchor[0]),
                    "y1": float(anchor[1]),
                    "x2": float(anchor[2]),
                    "y2": float(anchor[3]),
                    "label": anchor_label or "Place here",
                    "anchor": True,
                }
            )
        payload = {"boxes": boxes}
        try:
            await ui.run_javascript(
                "return window.__soCam.drawOverlays("
                + json.dumps(payload)
                + ")",
                timeout=3.0,
            )
        except Exception as exc:
            logger.debug("draw overlays: %s", exc)

    def refresh_steps() -> None:
        step_row.clear()
        idx = STEPS.index(state.step)
        with step_row:
            for i, key in enumerate(STEPS):
                cls = "so-step" + (
                    " is-done" if i < idx else " is-current" if i == idx else ""
                )
                with ui.element("div").classes(cls):
                    ui.html(f'<span class="so-step-num">{i + 1}</span>')
                    ui.label(STEP_LABELS[key])

    def show_step(step: str) -> None:
        state.step = step
        refresh_steps()
        for key, panel in step_panels.items():
            panel.set_visibility(key == step)

    def sync_cam_btn() -> None:
        text = "Stop camera" if state.live_running else "Start camera"
        color = "negative" if state.live_running else "primary"
        for btn in (cam_btn, place_cam_btn):
            try:
                btn.set_text(text)
                btn.props(f"color={color} unelevated no-caps")
            except Exception:
                pass

    def render_pick_list() -> None:
        pick_list.clear()
        with pick_list:
            if state.last_detections:
                ui.label("Detected — tap one, or use click / drag on the image").classes(
                    "so-muted mb-2"
                )
                for i, det in enumerate(state.last_detections):
                    name = str(det.get("label", "object")).capitalize()

                    def make(idx: int = i, n: str = name, box: list[float] = det["bbox"]) -> Any:
                        async def _() -> None:
                            label_input.set_value(n)
                            await confirm_selection(
                                bbox=list(box),
                                label_override=n,
                                from_detection=True,
                            )

                        return _

                    btn = (
                        ui.button(on_click=make())
                        .props("flat no-caps")
                        .classes("so-item w-full")
                    )
                    with btn:
                        ui.html('<span class="so-item-dot"></span>')
                        ui.label(name).classes("text-sm")
            else:
                ui.label(
                    "No auto-detections — click or drag on the image, then add a label."
                ).classes("so-muted")

    def render_place_summary() -> None:
        place_summary.clear()
        with place_summary:
            if state.selected_label:
                ui.html('<span class="so-pill">Selected</span>')
                ui.label(state.selected_label).classes("text-lg font-bold")
                if state.crop_preview_b64:
                    ui.image(
                        f"data:image/jpeg;base64,{state.crop_preview_b64}"
                    ).classes("w-20 h-20 rounded-2xl object-cover")

    def render_result() -> None:
        result_card.clear()
        if not state.selected_label and not state.placement:
            return
        with result_card:
            with ui.element("div").classes("so-result flex flex-col gap-2"):
                if state.selected_label:
                    ui.html('<span class="so-pill">Item</span>')
                    ui.label(state.selected_label).classes("text-2xl font-bold")
                    if state.crop_preview_b64:
                        ui.image(
                            f"data:image/jpeg;base64,{state.crop_preview_b64}"
                        ).classes("w-24 h-24 rounded-2xl object-cover mt-1")
                if state.placement:
                    ui.separator().classes("my-2")
                    ui.html('<span class="so-pill">Place here</span>')
                    ui.label(state.placement.get("target", "Open space")).classes(
                        "text-xl font-bold"
                    )
                    rec = state.placement.get("recommendation")
                    if rec:
                        ui.label(rec).classes("so-muted")

    async def ingest_snapshot_bgr(bgr: np.ndarray) -> None:
        set_error(None)
        set_status("Detecting items…")
        conf = float(state.detection_strength)
        pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        state.img_w, state.img_h = pil.size

        def work() -> tuple[list[dict[str, Any]], np.ndarray]:
            dets = run_detections_for_ui(pil, conf)
            return dets, draw_detections(bgr, dets)

        dets, annotated = await asyncio.to_thread(work)
        state.object_bgr = bgr
        state.last_detections = dets
        state.selected_label = None
        state.session_id = None
        state.crop_preview_b64 = None
        state.placement = None
        state.draft_bbox = None
        state.drag_start = None
        url = bgr_to_data_url(annotated, quality=88)
        show_freeze(url)
        freeze_img.set_content("")
        label_input.value = ""
        render_pick_list()
        show_step("pick")
        set_status(
            f"{len(dets)} detection(s) — click / drag on the image or pick from the list"
        )

    async def confirm_selection(
        *,
        bbox: list[float] | None = None,
        click: tuple[float, float] | None = None,
        label_override: str | None = None,
        from_detection: bool = False,
    ) -> None:
        if state.object_bgr is None:
            set_error("No snapshot yet.")
            return
        set_error(None)
        set_status("Locking selection…")
        bgr = state.object_bgr
        conf = float(state.detection_strength)
        pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        custom = (label_override or label_input.value or "").strip()

        def work() -> dict[str, Any]:
            detections = run_detections_for_ui(pil, conf)
            selected = None
            crop_bbox: list[float]

            if bbox is not None:
                crop_bbox = vision._clamp_bbox(list(bbox))
                cx = (crop_bbox[0] + crop_bbox[2]) / 2
                cy = (crop_bbox[1] + crop_bbox[3]) / 2
                selected = vision._select_detection_by_point(detections, cx, cy)
            elif click is not None:
                selected = vision._select_detection_by_point(detections, click[0], click[1])
                if selected:
                    crop_bbox = selected["bbox"]
                else:
                    pad = 0.1
                    crop_bbox = vision._clamp_bbox(
                        [click[0] - pad, click[1] - pad, click[0] + pad, click[1] + pad]
                    )
            else:
                return {"error": "Click a point or draw a box first."}

            crop_bbox = vision._try_refine_bbox_with_seg(pil, crop_bbox)
            crop = vision._crop_bgr(bgr, crop_bbox)

            if custom:
                label, confidence = custom, 1.0
            elif selected:
                label, confidence = selected["label"], float(selected["confidence"])
                crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                crop_dets = vision._run_detections(crop_pil, conf=max(0.15, conf * 0.7))
                if crop_dets and crop_dets[0]["confidence"] >= 0.3 and not from_detection:
                    label, confidence = crop_dets[0]["label"], crop_dets[0]["confidence"]
            else:
                label, confidence = "object", 0.4

            sid, crop_b64 = create_session(label, confidence, crop_bbox, crop, pil.size)
            annotated = draw_detections(
                bgr,
                detections,
                selected_index=next(
                    (
                        i
                        for i, d in enumerate(detections)
                        if d.get("bbox") == (selected or {}).get("bbox")
                    ),
                    None,
                ),
            )
            # ensure selection box visible
            from bbox_draw import draw_placement_anchor as _box

            annotated = _box(annotated, crop_bbox, title=label, guidance=None)
            return {
                "label": label,
                "session_id": sid,
                "crop_b64": crop_b64,
                "annotated": annotated,
                "bbox": crop_bbox,
            }

        result = await asyncio.to_thread(work)
        if "error" in result:
            set_error(result["error"])
            return

        state.selected_label = result["label"]
        state.session_id = result["session_id"]
        state.crop_preview_b64 = result["crop_b64"]
        show_freeze(bgr_to_data_url(result["annotated"], quality=88))
        render_place_summary()
        render_result()
        show_step("place")
        set_status("Find similar items in storage — upload a shelf photo or snapshot")
        ui.notify(f"Selected: {result['label']}", type="positive")

    async def suggest_placement(bgr: np.ndarray) -> None:
        if not state.selected_label:
            set_error("Select an item first.")
            return
        if state._infer_busy:
            return
        state._infer_busy = True
        set_error(None)
        set_status("Looking for similar items…")
        conf = float(state.detection_strength)
        label = state.selected_label
        pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

        def work() -> dict[str, Any]:
            dets = run_detections_for_ui(pil, conf)
            place = suggest_similar_placement(label or "object", dets, min_conf=conf)
            annotated = draw_detections(bgr, dets)
            annotated = draw_placement_anchor(
                annotated,
                place["anchor_box"],
                title=place["target"],
                guidance=place.get("guidance"),
            )
            place["annotated"] = annotated
            place["detections"] = dets
            return place

        try:
            result = await asyncio.to_thread(work)
            state.place_bgr = bgr
            state.placement = result
            show_freeze(bgr_to_data_url(result["annotated"], quality=85))
            clear_live_overlays()
            render_result()
            show_step("done")
            set_status("Placement ready")
        finally:
            state._infer_busy = False

    async def live_overlay_tick() -> None:
        """Draw detection / placement boxes on top of the live video (no JPEG swap)."""
        if not state.live_running or state._infer_busy:
            return
        # Photo step: always show live detections
        # Place/done: show when placement overlay toggle is on
        want_detect = state.step == "photo"
        want_place = (
            state.placement_overlay
            and state.step in ("place", "done")
            and bool(state.selected_label)
        )
        if not want_detect and not want_place:
            return

        state._infer_busy = True
        try:
            data_url = await ui.run_javascript(
                "return window.__soCam && window.__soCam.snapshot()",
                timeout=5.0,
            )
            if not data_url:
                return
            bgr = await asyncio.to_thread(data_url_to_bgr, data_url)
            conf = float(state.detection_strength)

            def work() -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
                pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                dets = run_detections_for_ui(pil, conf)
                place = None
                if want_place:
                    place = suggest_similar_placement(
                        state.selected_label or "object", dets, min_conf=conf
                    )
                return dets, place

            dets, place = await asyncio.to_thread(work)
            if want_detect:
                state.last_detections = dets
            if place:
                state.placement = place
                await draw_live_overlays(
                    detections=dets if want_detect else None,
                    anchor=place["anchor_box"],
                    anchor_label=place.get("target"),
                )
            else:
                await draw_live_overlays(detections=dets)
            show_live(True)
            if want_detect and state.step == "photo":
                n = len(dets)
                set_status(
                    f"Camera on — {n} detection{'s' if n != 1 else ''} · snapshot when ready"
                )
        except Exception as exc:
            logger.debug("live overlay tick: %s", exc)
        finally:
            state._infer_busy = False

    async def start_live() -> None:
        set_error(None)
        device_id = cam_select.value or ""
        try:
            ok = await ui.run_javascript(
                f"return await window.__soCam.start({device_id!r} || null)",
                timeout=15.0,
            )
        except Exception as exc:
            set_error(f"Camera error: {exc}")
            return
        if not ok:
            set_error("Could not start camera. Check permissions.")
            return
        state.live_running = True
        sync_cam_btn()
        show_live(True)
        set_status("Camera on — detecting… snapshot when ready")
        # Timer is created during page build — only activate here (never create UI after await)
        if place_timer is not None:
            place_timer.activate()

    async def stop_live() -> None:
        state.live_running = False
        if place_timer is not None:
            place_timer.deactivate()
        try:
            await ui.run_javascript("window.__soCam.stop()", timeout=5.0)
        except Exception:
            pass
        sync_cam_btn()
        clear_live_overlays()
        if state.freeze_url:
            show_freeze(state.freeze_url)
        else:
            show_live(False)
            clear_preview()
        set_status("Camera stopped")

    async def toggle_camera() -> None:
        if state.live_running:
            await stop_live()
        else:
            await start_live()

    async def take_snapshot() -> None:
        if not state.live_running:
            set_error("Start the camera first, or upload a photo.")
            return
        data_url = await ui.run_javascript(
            "return window.__soCam.snapshot()",
            timeout=5.0,
        )
        if not data_url:
            set_error("No frame yet — wait a moment after starting the camera.")
            return
        await stop_live()
        bgr = await asyncio.to_thread(data_url_to_bgr, data_url)
        await ingest_snapshot_bgr(bgr)

    async def on_upload(e: Any) -> None:
        if state.live_running:
            await stop_live()
        set_error(None)
        pil = pil_from_bytes(read_upload_bytes(e))
        await ingest_snapshot_bgr(pil_to_bgr(pil))

    async def on_storage_upload(e: Any) -> None:
        if state.live_running:
            await stop_live()
        await suggest_placement(pil_to_bgr(pil_from_bytes(read_upload_bytes(e))))

    async def snapshot_for_placement() -> None:
        if not state.live_running:
            set_error("Start the camera, then snapshot the shelf — or upload a photo.")
            return
        data_url = await ui.run_javascript(
            "return window.__soCam.snapshot()",
            timeout=5.0,
        )
        if not data_url:
            set_error("No camera frame.")
            return
        if not state.placement_overlay:
            await stop_live()
        bgr = await asyncio.to_thread(data_url_to_bgr, data_url)
        await suggest_placement(bgr)

    async def on_placement_toggle(e: Any) -> None:
        state.placement_overlay = bool(e.value)
        if state.placement_overlay:
            set_status("Overlay on — aim at storage (updates every ~2s, video stays smooth)")
            if not state.live_running and state.step in ("place", "done"):
                await start_live()
        else:
            clear_live_overlays()
            set_status("Placement overlay off")

    def on_select_mouse(e: Any) -> None:
        if state.step != "pick" or state.object_bgr is None:
            return
        ix = float(getattr(e, "image_x", 0) or 0)
        iy = float(getattr(e, "image_y", 0) or 0)
        w = max(1.0, float(state.img_w))
        h = max(1.0, float(state.img_h))
        # NiceGUI reports natural image pixel coords
        if ix <= 1.5 and iy <= 1.5 and w > 2 and h > 2:
            # already normalized (some builds)
            nx, ny = min(1.0, max(0.0, ix)), min(1.0, max(0.0, iy))
            px, py = nx * w, ny * h
        else:
            nx, ny = min(1.0, max(0.0, ix / w)), min(1.0, max(0.0, iy / h))
            px, py = ix, iy

        etype = str(getattr(e, "type", "") or "")

        if etype == "mousedown":
            state.drag_start = (nx, ny)
            state.draft_bbox = None
            freeze_img.set_content(
                f'<circle cx="{px:.1f}" cy="{py:.1f}" r="7" fill="none" '
                f'stroke="#22d3ee" stroke-width="2"/>'
            )
        elif etype == "mousemove" and state.drag_start:
            x0, y0 = state.drag_start
            x1, x2 = sorted((x0, nx))
            y1, y2 = sorted((y0, ny))
            if (x2 - x1) > 0.01 and (y2 - y1) > 0.01:
                state.draft_bbox = [x1, y1, x2, y2]
                freeze_img.set_content(
                    f'<rect x="{x1*w:.1f}" y="{y1*h:.1f}" width="{(x2-x1)*w:.1f}" '
                    f'height="{(y2-y1)*h:.1f}" fill="rgba(34,211,238,0.15)" '
                    f'stroke="#22d3ee" stroke-width="2" rx="10"/>'
                )
        elif etype == "mouseup" and state.drag_start:
            x0, y0 = state.drag_start
            state.drag_start = None
            x1, x2 = sorted((x0, nx))
            y1, y2 = sorted((y0, ny))
            if (x2 - x1) > 0.02 and (y2 - y1) > 0.02:
                state.draft_bbox = [x1, y1, x2, y2]
                freeze_img.set_content(
                    f'<rect x="{x1*w:.1f}" y="{y1*h:.1f}" width="{(x2-x1)*w:.1f}" '
                    f'height="{(y2-y1)*h:.1f}" fill="rgba(34,211,238,0.18)" '
                    f'stroke="#22d3ee" stroke-width="2" rx="10"/>'
                )
                set_status("Box selected — optional label, then Confirm")
            else:
                state.draft_bbox = [nx, ny, nx, ny]
                freeze_img.set_content(
                    f'<circle cx="{px:.1f}" cy="{py:.1f}" r="11" fill="none" '
                    f'stroke="#22d3ee" stroke-width="2"/>'
                    f'<line x1="{px-16:.1f}" y1="{py:.1f}" x2="{px+16:.1f}" y2="{py:.1f}" '
                    f'stroke="#22d3ee" stroke-width="2"/>'
                    f'<line x1="{px:.1f}" y1="{py-16:.1f}" x2="{px:.1f}" y2="{py+16:.1f}" '
                    f'stroke="#22d3ee" stroke-width="2"/>'
                )
                set_status("Point selected — optional label, then Confirm")

    async def confirm_from_ui() -> None:
        if not state.draft_bbox:
            set_error("Click or drag on the image first (or pick a detection).")
            return
        x1, y1, x2, y2 = state.draft_bbox
        if abs(x2 - x1) < 0.015 and abs(y2 - y1) < 0.015:
            await confirm_selection(click=(x1, y1), label_override=label_input.value)
        else:
            await confirm_selection(bbox=[x1, y1, x2, y2], label_override=label_input.value)

    async def reset_all() -> None:
        if state.live_running:
            await stop_live()
        state.session_id = None
        state.selected_label = None
        state.crop_preview_b64 = None
        state.last_detections = []
        state.placement = None
        state.object_bgr = None
        state.place_bgr = None
        state.placement_overlay = False
        state.draft_bbox = None
        place_toggle.value = False
        label_input.value = ""
        clear_preview()
        pick_list.clear()
        result_card.clear()
        place_summary.clear()
        set_error(None)
        set_status("Take a photo or upload one to begin")
        show_step("photo")

    async def refresh_cameras() -> None:
        try:
            devices = await ui.run_javascript(
                "return await window.__soCam.list()",
                timeout=15.0,
            )
            opts: dict[str, str] = {"": "Default camera"}
            if isinstance(devices, list):
                for d in devices:
                    opts[str(d.get("id", ""))] = str(d.get("label") or "Camera")
            cam_select.options = opts
            cam_select.update()
        except Exception as exc:
            logger.warning("enumerate cameras: %s", exc)

    # ---- layout ----
    with ui.element("div").classes("so-shell") as page_root:
        render_header(active="organize")
        with ui.element("div").classes("so-main"):
            ui.label(
                "Snapshot an item, click or drag to select it, optionally name it, "
                "then we’ll place it with similar items — no made-up bin names."
            ).classes("so-sub")

            step_row = ui.element("div").classes("so-steps")
            refresh_steps()
            status_label = ui.label(state.status).classes("so-muted")
            error_label = ui.label("").classes("so-error")
            error_label.set_visibility(False)

            with ui.element("div").classes("so-grid"):
                with ui.element("div").classes("so-panel"):
                    # PHOTO
                    photo_panel = ui.column().classes("w-full gap-3")
                    step_panels["photo"] = photo_panel
                    with photo_panel:
                        ui.label("1 · Photo").classes("so-panel-title")
                        ui.label("Start the webcam (smooth live feed) or upload.").classes(
                            "so-muted"
                        )
                        cam_select = ui.select(
                            camera_options, value="", label="Camera"
                        ).props("filled dense hide-bottom-space").classes("w-full")
                        ui.button(
                            "Refresh cameras",
                            on_click=refresh_cameras,
                        ).props("flat dense no-caps")
                        cam_btn = (
                            ui.button(
                                "Start camera",
                                on_click=toggle_camera,
                            )
                            .props("color=primary unelevated no-caps")
                            .classes("w-full")
                        )
                        ui.button(
                            "Take snapshot",
                            on_click=take_snapshot,
                        ).props("outline no-caps").classes("w-full")
                        ui.upload(
                            label="Or upload a photo",
                            auto_upload=True,
                            on_upload=on_upload,
                        ).props('accept="image/*" flat bordered').classes("w-full so-upload")
                        ui.separator()
                        ui.label("Detection strength").classes(
                            "text-xs font-bold uppercase tracking-wide"
                        ).style("color: var(--so-muted)")
                        strength = (
                            ui.slider(
                                min=0.1, max=0.85, step=0.05, value=state.detection_strength
                            )
                            .props("label label-always color=cyan")
                            .classes("w-full")
                        )
                        strength.on_value_change(
                            lambda e: setattr(state, "detection_strength", float(e.value))
                        )
                        ui.label("Lower finds more items; higher is stricter.").classes(
                            "so-muted"
                        )

                    # PICK
                    pick_panel = ui.column().classes("w-full gap-3")
                    step_panels["pick"] = pick_panel
                    pick_panel.set_visibility(False)
                    with pick_panel:
                        ui.label("2 · Select item").classes("so-panel-title")
                        ui.label(
                            "Click = point · click-drag = box. Add your own label if you want "
                            "(used when the item is unknown)."
                        ).classes("so-muted")
                        label_input = ui.input(
                            label="Your label (optional)",
                            placeholder="e.g. water bottle, charger…",
                        ).props("filled dense hide-bottom-space").classes("w-full")
                        pick_list = ui.column().classes("w-full gap-2")
                        ui.button(
                            "Confirm selection",
                            on_click=confirm_from_ui,
                        ).props("color=primary unelevated no-caps").classes("w-full")
                        with ui.row().classes("w-full gap-2 flex-wrap"):
                            ui.button(
                                "Back", on_click=lambda: show_step("photo")
                            ).props("flat no-caps")
                            ui.button(
                                "Retake",
                                on_click=reset_all,
                            ).props("outline no-caps")

                    # PLACE
                    place_panel = ui.column().classes("w-full gap-3")
                    step_panels["place"] = place_panel
                    place_panel.set_visibility(False)
                    with place_panel:
                        ui.label("3 · Find spot").classes("so-panel-title")
                        place_summary = ui.column().classes("w-full gap-2 mb-1")
                        ui.label(
                            "We’ll look for similar items in the scene and suggest a spot beside them."
                        ).classes("so-muted")
                        place_toggle = ui.switch(
                            "Show live placement overlay",
                            value=False,
                            on_change=on_placement_toggle,
                        )
                        place_cam_btn = (
                            ui.button(
                                "Start camera",
                                on_click=toggle_camera,
                            )
                            .props("outline unelevated no-caps")
                            .classes("w-full")
                        )
                        ui.button(
                            "Snapshot shelf",
                            on_click=snapshot_for_placement,
                        ).props("color=primary unelevated no-caps").classes("w-full")
                        ui.upload(
                            label="Or upload shelf / storage photo",
                            auto_upload=True,
                            on_upload=on_storage_upload,
                        ).props('accept="image/*" flat bordered').classes("w-full so-upload")
                        ui.button(
                            "Back to select", on_click=lambda: show_step("pick")
                        ).props("flat no-caps")

                    # DONE
                    done_panel = ui.column().classes("w-full gap-3")
                    step_panels["done"] = done_panel
                    done_panel.set_visibility(False)
                    with done_panel:
                        ui.label("4 · Done").classes("so-panel-title")
                        ui.label(
                            "Put the item in the highlighted area — near similar things when found."
                        ).classes("so-muted")
                        result_card = ui.column().classes("w-full gap-2")
                        ui.button(
                            "Organize another",
                            on_click=reset_all,
                        ).props("color=primary unelevated no-caps").classes("w-full")

                # VIEW
                with ui.element("div").classes("so-panel"):
                    ui.label("View").classes("so-panel-title")
                    with ui.element("div").classes("so-frame is-empty") as wrap:
                        frame_wrap = wrap
                        live_video = (
                            ui.element("video")
                            .props("id=so-live-video autoplay playsinline muted")
                            .classes("w-full h-full")
                        )
                        live_video.set_visibility(False)
                        freeze_img = ui.interactive_image(
                            cross=False,
                            events=["mousedown", "mouseup", "mousemove"],
                        ).classes("w-full h-full")
                        freeze_img.on_mouse(on_select_mouse)
                        freeze_img.set_visibility(False)
                        overlay_layer = (
                            ui.element("div")
                            .props("id=so-overlay-layer")
                            .classes("so-overlay-layer")
                        )

    # Timers must be created during page build (not after await)
    place_timer = ui.timer(0.9, live_overlay_tick, active=False)
    ui.timer(0.4, refresh_cameras, once=True)

    async def _shutdown() -> None:
        try:
            await ui.run_javascript("window.__soCam && window.__soCam.stop()", timeout=3.0)
        except Exception:
            pass

    nicegui_app.on_disconnect(_shutdown)


def run() -> None:
    ui.run(
        title="Smart Organizer",
        host="0.0.0.0",
        port=8080,
        reload=False,
        show=True,
        favicon="📦",
        dark=None,
    )


if __name__ in {"__main__", "__mp_main__"}:
    run()
