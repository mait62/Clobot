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


def downscale_bgr(bgr: np.ndarray, max_side: int = 1280) -> np.ndarray:
    """Keep mobile snapshots small enough for YOLO + NiceGUI image updates."""
    h, w = bgr.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return bgr
    scale = max_side / float(m)
    return cv2.resize(
        bgr,
        (max(1, int(w * scale)), max(1, int(h * scale))),
        interpolation=cv2.INTER_AREA,
    )


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
        self._overlay_busy = False
        self._action_busy = False
        self.active_facing: str | None = None  # 'user' | 'environment'
        self.manual_select = False
        # Selection draft (normalized)
        self.drag_start: tuple[float, float] | None = None
        self.draft_bbox: list[float] | None = None
        self.img_w = 1
        self.img_h = 1



# Browser camera helpers — no server frame pumping
CAM_JS = """
window.__soCamList = [];
window.__soPreferredCam = null;
window.__soCam = {
  stream: null,
  isIOS() {
    const ua = navigator.userAgent || '';
    return /iPad|iPhone|iPod/.test(ua) ||
      (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1);
  },
  classifyLabel(label) {
    const l = String(label || '').toLowerCase();
    if (/front|user|face|selfie/.test(l)) return 'user';
    if (/back|rear|environment|world|triple|dual|wide|ultra|tele/.test(l)) return 'environment';
    return null;
  },
  async list() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) return [];
    try {
      const all = await navigator.mediaDevices.enumerateDevices();
      const cams = all.filter(d => d.kind === 'videoinput');
      const mapped = cams.map((d, i) => {
        const label = d.label || ('Camera ' + (i + 1));
        return {
          id: d.deviceId || '',
          label,
          facing: this.classifyLabel(label),
        };
      }).filter(d => d.id);
      window.__soCamList = mapped;
      return mapped;
    } catch (_) {
      return [];
    }
  },
  facingSummary(devices) {
    const list = devices || window.__soCamList || [];
    const hasFront = list.some(d => d.facing === 'user');
    const hasBack = list.some(d => d.facing === 'environment');
    return {
      count: list.length,
      hasFront,
      hasBack,
      // iOS/iPad always supports facingMode even before labels appear
      showFront: hasFront || this.isIOS(),
      showBack: hasBack || this.isIOS(),
      devices: list,
    };
  },
  async start(opts) {
    opts = opts || {};
    this.stop();
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error('Camera API not available. Use HTTPS Safari/Chrome.');
    }
    let facing = opts.facing || null;
    let deviceId = opts.deviceId || null;
    if (deviceId && !facing) {
      const info = (window.__soCamList || []).find(d => d.id === deviceId);
      if (info && info.facing) facing = info.facing;
    }
    const attempts = [];
    // Prefer explicit device when the user picked one (works on iPad too)
    if (deviceId) {
      attempts.push({ video: { deviceId: { exact: deviceId } }, audio: false });
      attempts.push({ video: { deviceId: { ideal: deviceId } }, audio: false });
      if (facing) {
        attempts.push({
          video: { deviceId: { ideal: deviceId }, facingMode: { ideal: facing } },
          audio: false,
        });
      }
    }
    if (facing) {
      attempts.push({ video: { facingMode: { exact: facing } }, audio: false });
      attempts.push({ video: { facingMode: facing }, audio: false });
      attempts.push({ video: { facingMode: { ideal: facing } }, audio: false });
    }
    attempts.push({ video: { facingMode: { ideal: 'environment' } }, audio: false });
    attempts.push({ video: true, audio: false });

    let stream = null;
    let lastErr = null;
    for (const constraints of attempts) {
      try {
        stream = await navigator.mediaDevices.getUserMedia(constraints);
        break;
      } catch (err) {
        lastErr = err;
      }
    }
    if (!stream) {
      throw lastErr || new Error('Could not open camera');
    }
    this.stream = stream;
    const track = stream.getVideoTracks()[0];
    const settings = (track && track.getSettings) ? track.getSettings() : {};
    const activeFacing = settings.facingMode || facing || this.classifyLabel(track && track.label) || null;
    const activeId = settings.deviceId || deviceId || null;
    if (activeId) window.__soPreferredCam = activeId;

    const v = document.getElementById('so-live-video');
    if (v) {
      v.setAttribute('playsinline', 'true');
      v.setAttribute('webkit-playsinline', 'true');
      v.muted = true;
      v.autoplay = true;
      v.srcObject = stream;
      try { await v.play(); } catch (_) {}
    }
    // Refresh labels now that permission was granted
    try { await this.list(); } catch (_) {}
    return {
      ok: true,
      facing: activeFacing,
      deviceId: activeId,
      label: (track && track.label) || null,
    };
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
    facing_btn_refs: list[tuple[ui.button, str]] = []
    facing_row: ui.element | None = None
    place_facing_row: ui.element | None = None
    front_btn_wrap: ui.element | None = None
    back_btn_wrap: ui.element | None = None
    m_back: ui.button | None = None
    m_front: ui.button | None = None

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
    confirm_btn: ui.button
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

    def sync_facing_btns() -> None:
        """Highlight the active Front/Back control; clear when using device picker."""
        for btn, facing in facing_btn_refs:
            try:
                active = bool(state.live_running and state.active_facing == facing)
                if active:
                    btn.props(
                        remove="outline flat",
                        add="color=primary unelevated no-caps",
                    )
                else:
                    btn.props(
                        remove="unelevated",
                        add="outline color=primary no-caps",
                    )
            except Exception:
                pass

    def detection_at(nx: float, ny: float) -> dict[str, Any] | None:
        """Return the smallest detection whose box contains the normalized point."""
        hits: list[tuple[float, dict[str, Any]]] = []
        for det in state.last_detections:
            box = det.get("bbox") or []
            if len(box) != 4:
                continue
            x1, y1, x2, y2 = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
            if x1 <= nx <= x2 and y1 <= ny <= y2:
                area = max(1e-6, (x2 - x1) * (y2 - y1))
                hits.append((area, det))
        if not hits:
            return None
        hits.sort(key=lambda t: t[0])
        return hits[0][1]

    def render_pick_list() -> None:
        pick_list.clear()
        with pick_list:
            if state.last_detections:
                hint = (
                    "Detected — tap one below, or click it on the image"
                    if not state.manual_select
                    else "Detected — tap one, or draw a box on the image (manual)"
                )
                ui.label(hint).classes("so-muted mb-2")
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
                if state.manual_select:
                    ui.label(
                        "No auto-detections — draw a box on the image, then add a label."
                    ).classes("so-muted")
                else:
                    ui.label(
                        "No auto-detections — turn on Manual selection to draw a box."
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
        if state.manual_select:
            set_status(
                f"{len(dets)} detection(s) — draw a box or pick from the list, then Confirm"
            )
        else:
            set_status(
                f"{len(dets)} detection(s) — click an item on the image (or the list)"
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
        if state._action_busy:
            set_status("Still finishing the last placement…")
            return
        state._action_busy = True
        set_error(None)
        set_status("Looking for similar items…")
        conf = float(state.detection_strength)
        label = state.selected_label
        bgr = downscale_bgr(bgr)
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
            show_freeze(bgr_to_data_url(result["annotated"], quality=78))
            clear_live_overlays()
            render_result()
            show_step("done")
            set_status("Placement ready — see the highlighted spot")
            ui.notify("Placement ready", type="positive")
            # On phones the sticky preview can hide the Done panel — scroll it into view
            try:
                await ui.run_javascript(
                    """
const el = document.querySelector('.so-grid-controls');
if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
""",
                    timeout=2.0,
                )
            except Exception:
                pass
        except Exception as exc:
            logger.exception("placement failed")
            set_error(f"Placement failed: {exc}")
            set_status("Could not finish placement — try again or upload a photo")
        finally:
            state._action_busy = False

    async def live_overlay_tick() -> None:
        """Draw detection / placement boxes on top of the live video (no JPEG swap)."""
        if not state.live_running or state._overlay_busy or state._action_busy:
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

        state._overlay_busy = True
        try:
            data_url = await ui.run_javascript(
                "return window.__soCam && window.__soCam.snapshot()",
                timeout=5.0,
            )
            if not data_url:
                return
            bgr = await asyncio.to_thread(data_url_to_bgr, data_url)
            bgr = downscale_bgr(bgr, max_side=960)
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
            state._overlay_busy = False

    async def on_cam_client_result(e: Any) -> None:
        """Camera open runs in the browser (iOS needs the tap gesture); Python only syncs UI."""
        args = getattr(e, "args", e)
        if isinstance(args, list) and args:
            args = args[0]
        if not isinstance(args, dict):
            args = {}
        action = args.get("action") or "start"
        if action == "stop":
            await stop_live()
            return
        if action == "prefer":
            # Dropdown changed while camera is off — just remember the choice
            did = args.get("deviceId") or ""
            try:
                cam_select.value = did
                cam_select.update()
            except Exception:
                pass
            return
        if not args.get("ok"):
            msg = str(args.get("message") or "Could not start camera")
            low = msg.lower()
            if "notallowed" in low or "permission" in low:
                set_error(
                    "Camera blocked. On iPad/iPhone: use HTTPS, tap Allow, "
                    "then pick Front / Back or a camera in the list."
                )
            elif "secure" in low or "https" in low or "mediadevices" in low:
                set_error(
                    "iOS needs HTTPS for camera. Use Pinggy/ngrok https://… — not http://IP."
                )
            else:
                set_error(f"Camera error: {msg}")
            return
        set_error(None)
        state.live_running = True
        facing = args.get("facing")
        if facing in ("user", "environment"):
            state.active_facing = str(facing)
        else:
            state.active_facing = None
        device_id = args.get("deviceId") or ""
        if device_id:
            try:
                await ui.run_javascript(
                    "window.__soSuppressCamSelect = true;"
                    "setTimeout(() => { window.__soSuppressCamSelect = false; }, 400);"
                )
                if cam_select.value != device_id:
                    cam_select.value = device_id
                    cam_select.update()
            except Exception:
                pass
        sync_cam_btn()
        sync_facing_btns()
        show_live(True)
        cam_label = args.get("label") or (
            "Front camera"
            if state.active_facing == "user"
            else "Back camera"
            if state.active_facing == "environment"
            else "Camera"
        )
        set_status(f"{cam_label} on — detecting… snapshot when ready")
        await refresh_cameras()
        if place_timer is not None:
            place_timer.activate()

    def _cam_start_js(facing: str | None, *, prefer_device: bool = False) -> str:
        """Open camera in-page so iOS keeps the user gesture (no Python round-trip first)."""
        facing_js = json.dumps(facing)
        if prefer_device:
            return """
async () => {
  if (window.__soCam && window.__soCam.stream) { emit({action: 'stop'}); return; }
  try {
    const deviceId = window.__soPreferredCam || null;
    const info = (window.__soCamList || []).find(d => d.id === deviceId);
    const facing = (info && info.facing) || null;
    const result = await window.__soCam.start({deviceId: deviceId, facing: facing});
    emit(Object.assign({ok: true, action: 'start'}, result || {}));
  } catch (err) {
    emit({ok: false, action: 'start', message: String((err && err.message) || err)});
  }
}
""".strip()
        return f"""
async () => {{
  try {{
    const result = await window.__soCam.start({{facing: {facing_js}, deviceId: null}});
    emit(Object.assign({{ok: true, action: 'start', facing: {facing_js}}}, result || {{}}));
  }} catch (err) {{
    emit({{ok: false, action: 'start', message: String((err && err.message) || err)}});
  }}
}}
""".strip()

    CAM_SELECT_JS = """
async (value) => {
  const deviceId = value || null;
  window.__soPreferredCam = deviceId;
  if (window.__soSuppressCamSelect) {
    emit({action: 'prefer', deviceId: deviceId});
    return;
  }
  if (!(window.__soCam && window.__soCam.stream)) {
    emit({action: 'prefer', deviceId: deviceId});
    return;
  }
  try {
    const info = (window.__soCamList || []).find(d => d.id === deviceId);
    const facing = (info && info.facing) || null;
    const result = await window.__soCam.start({deviceId: deviceId, facing: facing});
    emit(Object.assign({ok: true, action: 'start'}, result || {}, {deviceId: deviceId}));
  } catch (err) {
    emit({ok: false, action: 'start', message: String((err && err.message) || err)});
  }
}
""".strip()

    async def stop_live() -> None:
        state.live_running = False
        state.active_facing = None
        if place_timer is not None:
            place_timer.deactivate()
        try:
            await ui.run_javascript("window.__soCam.stop()", timeout=5.0)
        except Exception:
            pass
        sync_cam_btn()
        sync_facing_btns()
        clear_live_overlays()
        if state.freeze_url:
            show_freeze(state.freeze_url)
        else:
            show_live(False)
            clear_preview()
        set_status("Camera stopped")

    def wire_cam_button(
        btn: ui.button, facing: str | None = None, *, prefer_device: bool = False
    ) -> None:
        btn.on(
            "click",
            handler=on_cam_client_result,
            js_handler=_cam_start_js(facing, prefer_device=prefer_device),
        )
        if facing in ("user", "environment"):
            facing_btn_refs.append((btn, facing))

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
        bgr = downscale_bgr(bgr)
        await ingest_snapshot_bgr(bgr)

    async def on_upload(e: Any) -> None:
        if state.live_running:
            await stop_live()
        set_error(None)
        pil = pil_from_bytes(read_upload_bytes(e))
        await ingest_snapshot_bgr(downscale_bgr(pil_to_bgr(pil)))

    async def on_storage_upload(e: Any) -> None:
        if state.live_running:
            await stop_live()
        await suggest_placement(
            downscale_bgr(pil_to_bgr(pil_from_bytes(read_upload_bytes(e))))
        )

    async def snapshot_for_placement() -> None:
        if not state.live_running:
            set_error("Start the camera, then snapshot the shelf — or upload a photo.")
            return
        set_status("Capturing shelf…")
        data_url = await ui.run_javascript(
            "return window.__soCam.snapshot()",
            timeout=8.0,
        )
        if not data_url:
            set_error("No camera frame — wait a second after starting the camera.")
            return
        # Always stop live first so overlay ticks can't block / race the finish step
        await stop_live()
        for _ in range(40):
            if not state._overlay_busy:
                break
            await asyncio.sleep(0.05)
        try:
            bgr = await asyncio.to_thread(data_url_to_bgr, data_url)
        except Exception as exc:
            set_error(f"Could not read snapshot: {exc}")
            return
        await suggest_placement(bgr)

    async def smart_snapshot() -> None:
        """Mobile bar Snapshot: photo step vs shelf (place) step."""
        if state.step in ("place", "done"):
            await snapshot_for_placement()
        else:
            await take_snapshot()

    async def on_placement_toggle(e: Any) -> None:
        state.placement_overlay = bool(e.value)
        if state.placement_overlay:
            if not state.live_running and state.step in ("place", "done"):
                set_status(
                    "Overlay on — tap Back / Front camera so iOS can allow the stream"
                )
            else:
                set_status(
                    "Overlay on — aim at storage (updates ~every second, video stays smooth)"
                )
        else:
            clear_live_overlays()
            set_status("Placement overlay off")

    async def on_select_mouse(e: Any) -> None:
        if state.step != "pick" or state.object_bgr is None:
            return
        ix = float(getattr(e, "image_x", 0) or 0)
        iy = float(getattr(e, "image_y", 0) or 0)
        w = max(1.0, float(state.img_w))
        h = max(1.0, float(state.img_h))
        # NiceGUI reports natural image pixel coords
        if ix <= 1.5 and iy <= 1.5 and w > 2 and h > 2:
            nx, ny = min(1.0, max(0.0, ix)), min(1.0, max(0.0, iy))
            px, py = nx * w, ny * h
        else:
            nx, ny = min(1.0, max(0.0, ix / w)), min(1.0, max(0.0, iy / h))
            px, py = ix, iy

        etype = str(getattr(e, "type", "") or "")

        # Tap mode: click a detection to select it immediately
        if not state.manual_select:
            if etype == "mousedown":
                state.drag_start = (nx, ny)
                return
            if etype == "mouseup" and state.drag_start:
                state.drag_start = None
                det = detection_at(nx, ny)
                if not det:
                    set_error(None)
                    set_status(
                        "No detected item there — enable Manual selection to draw a box"
                    )
                    freeze_img.set_content(
                        f'<circle cx="{px:.1f}" cy="{py:.1f}" r="9" fill="none" '
                        f'stroke="#94a3b8" stroke-width="2"/>'
                    )
                    return
                name = str(det.get("label", "object")).capitalize()
                custom = (label_input.value or "").strip()
                label_input.set_value(custom or name)
                box = list(det["bbox"])
                x1, y1, x2, y2 = box
                freeze_img.set_content(
                    f'<rect x="{x1*w:.1f}" y="{y1*h:.1f}" width="{(x2-x1)*w:.1f}" '
                    f'height="{(y2-y1)*h:.1f}" fill="rgba(34,211,238,0.18)" '
                    f'stroke="#22d3ee" stroke-width="2" rx="10"/>'
                )
                await confirm_selection(
                    bbox=box,
                    label_override=custom or name,
                    from_detection=True,
                )
            return

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
                # Point in manual mode: prefer detection under cursor, else point crop
                det = detection_at(nx, ny)
                if det:
                    box = list(det["bbox"])
                    state.draft_bbox = box
                    name = str(det.get("label", "object")).capitalize()
                    if not (label_input.value or "").strip():
                        label_input.set_value(name)
                    x1, y1, x2, y2 = box
                    freeze_img.set_content(
                        f'<rect x="{x1*w:.1f}" y="{y1*h:.1f}" width="{(x2-x1)*w:.1f}" '
                        f'height="{(y2-y1)*h:.1f}" fill="rgba(34,211,238,0.18)" '
                        f'stroke="#22d3ee" stroke-width="2" rx="10"/>'
                    )
                    set_status(f"Selected {name} — Confirm when ready")
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
            summary = await ui.run_javascript(
                """
const devices = await window.__soCam.list();
return window.__soCam.facingSummary(devices);
""",
                timeout=15.0,
            )
            if not isinstance(summary, dict):
                summary = {}
            devices = summary.get("devices") or []
            opts: dict[str, str] = {"": "Default camera"}
            if isinstance(devices, list):
                for d in devices:
                    label = str(d.get("label") or "Camera")
                    facing = d.get("facing")
                    if facing == "user" and "front" not in label.lower():
                        label = f"{label} (Front)"
                    elif facing == "environment" and "back" not in label.lower():
                        label = f"{label} (Back)"
                    opts[str(d.get("id", ""))] = label
            cam_select.options = opts
            cur = cam_select.value or ""
            if cur and cur not in opts:
                cam_select.value = ""
            cam_select.update()

            show_front = bool(summary.get("showFront"))
            show_back = bool(summary.get("showBack"))
            # Only show Front/Back when we actually have (or iOS can use) those modes
            try:
                if front_btn_wrap is not None:
                    front_btn_wrap.set_visibility(show_front)
                if back_btn_wrap is not None:
                    back_btn_wrap.set_visibility(show_back)
                if facing_row is not None:
                    facing_row.set_visibility(show_front or show_back)
                if place_facing_row is not None:
                    place_facing_row.set_visibility(show_front or show_back)
                if m_back is not None:
                    m_back.set_visibility(show_back)
                if m_front is not None:
                    m_front.set_visibility(show_front)
            except Exception:
                pass
            sync_facing_btns()
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
                with ui.element("div").classes("so-panel so-grid-controls"):
                    # PHOTO
                    photo_panel = ui.column().classes("w-full gap-3")
                    step_panels["photo"] = photo_panel
                    with photo_panel:
                        ui.label("1 · Photo").classes("so-panel-title")
                        ui.label(
                            "Pick a camera in the list (switches live if already on), "
                            "or use Front / Back when available."
                        ).classes("so-muted")
                        cam_select = ui.select(
                            camera_options, value="", label="Camera"
                        ).props("filled dense hide-bottom-space").classes("w-full")
                        cam_select.on(
                            "update:model-value",
                            handler=on_cam_client_result,
                            js_handler=CAM_SELECT_JS,
                        )
                        with ui.row().classes(
                            "w-full gap-2 flex-wrap items-center"
                        ) as facing_row:
                            back_btn_wrap = ui.element("div").classes("flex-grow")
                            with back_btn_wrap:
                                back_btn = (
                                    ui.button("Back camera")
                                    .props("outline color=primary no-caps")
                                    .classes("w-full")
                                )
                                wire_cam_button(back_btn, "environment")
                            front_btn_wrap = ui.element("div").classes("flex-grow")
                            with front_btn_wrap:
                                front_btn = (
                                    ui.button("Front camera")
                                    .props("outline color=primary no-caps")
                                    .classes("w-full")
                                )
                                wire_cam_button(front_btn, "user")
                        # Hide until we know device labels / iOS
                        facing_row.set_visibility(False)
                        with ui.row().classes("w-full gap-2 flex-wrap items-center"):
                            ui.button(
                                "Refresh list",
                                on_click=refresh_cameras,
                            ).props("flat dense no-caps")
                            cam_btn = (
                                ui.button("Start / Stop")
                                .props("color=primary unelevated no-caps")
                                .classes("flex-grow")
                            )
                            wire_cam_button(cam_btn, prefer_device=True)
                        ui.button(
                            "Take snapshot",
                            on_click=take_snapshot,
                        ).props("outline no-caps").classes("w-full")
                        ui.upload(
                            label="Or upload a photo",
                            auto_upload=True,
                            on_upload=on_upload,
                        ).props('accept="image/*" flat bordered').classes("w-full so-upload")
                        with ui.element("div").classes("so-desktop-only w-full"):
                            ui.separator()
                            ui.label("Detection strength").classes(
                                "text-xs font-bold uppercase tracking-wide"
                            ).style("color: var(--so-muted)")
                            strength = (
                                ui.slider(
                                    min=0.1,
                                    max=0.85,
                                    step=0.05,
                                    value=state.detection_strength,
                                )
                                .props("label label-always color=cyan")
                                .classes("w-full")
                            )
                            strength.on_value_change(
                                lambda e: setattr(
                                    state, "detection_strength", float(e.value)
                                )
                            )
                            ui.label(
                                "Lower finds more items; higher is stricter."
                            ).classes("so-muted")
                        with ui.expansion("Detection strength", value=False).classes(
                            "w-full so-mobile-only"
                        ).props("dense"):
                            strength_m = (
                                ui.slider(
                                    min=0.1,
                                    max=0.85,
                                    step=0.05,
                                    value=state.detection_strength,
                                )
                                .props("label label-always color=cyan")
                                .classes("w-full")
                            )
                            strength_m.on_value_change(
                                lambda e: setattr(
                                    state, "detection_strength", float(e.value)
                                )
                            )

                    # PICK
                    pick_panel = ui.column().classes("w-full gap-3")
                    step_panels["pick"] = pick_panel
                    pick_panel.set_visibility(False)
                    with pick_panel:
                        ui.label("2 · Select item").classes("so-panel-title")
                        ui.label(
                            "Click a detected item on the image to select it. "
                            "Turn on Manual selection to draw your own box."
                        ).classes("so-muted")

                        def on_manual_toggle(e: Any) -> None:
                            state.manual_select = bool(e.value)
                            state.draft_bbox = None
                            state.drag_start = None
                            try:
                                freeze_img.set_content("")
                            except Exception:
                                pass
                            confirm_btn.set_visibility(state.manual_select)
                            render_pick_list()
                            if state.manual_select:
                                set_status("Manual on — drag a box, then Confirm")
                            else:
                                set_status("Tap a detected item on the image to select it")

                        ui.switch(
                            "Manual selection (draw box)",
                            value=False,
                            on_change=on_manual_toggle,
                        )
                        label_input = ui.input(
                            label="Your label (optional)",
                            placeholder="e.g. water bottle, charger…",
                        ).props("filled dense hide-bottom-space").classes("w-full")
                        pick_list = ui.column().classes("w-full gap-2")
                        confirm_btn = (
                            ui.button(
                                "Confirm selection",
                                on_click=confirm_from_ui,
                            )
                            .props("color=primary unelevated no-caps")
                            .classes("w-full")
                        )
                        confirm_btn.set_visibility(False)
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
                            ui.button("Start / Stop")
                            .props("outline unelevated no-caps")
                            .classes("w-full")
                        )
                        wire_cam_button(place_cam_btn, prefer_device=True)
                        with ui.row().classes(
                            "w-full gap-2 flex-wrap"
                        ) as place_facing_row:
                            place_back = (
                                ui.button("Back camera")
                                .props("outline color=primary no-caps")
                                .classes("flex-grow")
                            )
                            place_front = (
                                ui.button("Front camera")
                                .props("outline color=primary no-caps")
                                .classes("flex-grow")
                            )
                            wire_cam_button(place_back, "environment")
                            wire_cam_button(place_front, "user")
                        place_facing_row.set_visibility(False)
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
                with ui.element("div").classes("so-panel so-grid-view"):
                    ui.label("View").classes("so-panel-title")
                    with ui.element("div").classes("so-frame is-empty") as wrap:
                        frame_wrap = wrap
                        live_video = (
                            ui.element("video")
                            .props(
                                "id=so-live-video autoplay playsinline muted "
                                "webkit-playsinline"
                            )
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
                    with ui.element("div").classes("so-mobile-cam-bar so-mobile-only"):
                        m_back = (
                            ui.button("Back")
                            .props("outline color=primary no-caps dense")
                        )
                        m_front = (
                            ui.button("Front")
                            .props("outline color=primary no-caps dense")
                        )
                        m_snap = (
                            ui.button("Snapshot", on_click=smart_snapshot)
                            .props("outline unelevated no-caps dense")
                        )
                        wire_cam_button(m_back, "environment")
                        wire_cam_button(m_front, "user")
                        _ = m_snap

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
        viewport="width=device-width, initial-scale=1, maximum-scale=1, viewport-fit=cover",
    )


if __name__ in {"__main__", "__mp_main__"}:
    run()
