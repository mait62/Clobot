"""Backend settings page — /backend"""

from __future__ import annotations

import asyncio
from typing import Any

from nicegui import ui

import api_client
import app as vision
import settings as cfg
from ui_chrome import inject_styles, render_header


@ui.page("/backend")
def backend_page() -> None:
    inject_styles()
    data = cfg.load_settings()

    status_box = {"el": None}

    def set_msg(text: str, *, error: bool = False) -> None:
        color = "#e11d48" if error else "var(--so-muted)"
        if status_box["el"] is not None:
            status_box["el"].set_text(text)
            status_box["el"].style(f"color: {color}")

    def refresh_local_status() -> None:
        try:
            st = vision.model_status()
            local_status.set_text(
                f"Local model: {st['model']} · device={st['device']} · "
                f"loaded={st['model_loaded']} · conf={st['default_conf']} · imgsz={st['imgsz']}"
            )
        except Exception as exc:
            local_status.set_text(f"Local status unavailable: {exc}")

    async def test_remote() -> None:
        url = (api_url.value or "").strip()
        set_msg("Pinging remote API…")
        result = await asyncio.to_thread(api_client.health_check, url)
        if result["ok"]:
            payload = result.get("data") or {}
            set_msg(
                f"Remote OK — model={payload.get('model')} device={payload.get('device')} "
                f"status={payload.get('status')}"
            )
            ui.notify("Remote backend reachable", type="positive")
        else:
            set_msg(f"Remote failed: {result.get('error')}", error=True)
            ui.notify("Remote backend unreachable", type="negative")

    async def save_and_apply() -> None:
        model = (model_select.value or "").strip()
        if custom_model.value and custom_model.value.strip():
            model = custom_model.value.strip()
        try:
            conf = float(conf_slider.value)
            imgsz = int(imgsz_input.value or 640)
        except Exception:
            set_msg("Invalid conf / imgsz", error=True)
            return

        updates = {
            "api_base_url": (api_url.value or "").strip() or cfg.DEFAULTS["api_base_url"],
            "use_remote_api": bool(use_remote.value),
            "yolo_model": model or cfg.DEFAULTS["yolo_model"],
            "default_conf": conf,
            "imgsz": imgsz,
            "device": device_select.value or "auto",
        }
        cfg.save_settings(updates)
        set_msg("Saved. Applying to local engine…")
        try:
            applied = await asyncio.to_thread(cfg.apply_to_vision)
            set_msg(
                f"Applied — model={applied['yolo_model']} conf={applied['default_conf']} "
                f"remote={applied['use_remote_api']}"
            )
            ui.notify("Settings applied", type="positive")
            refresh_local_status()
        except Exception as exc:
            set_msg(f"Saved, but model reload failed: {exc}", error=True)
            ui.notify(str(exc), type="negative")

    with ui.element("div").classes("so-shell"):
        render_header(active="backend")
        with ui.element("div").classes("so-main"):
            ui.label("Backend & model settings").classes("text-2xl font-bold")
            ui.label(
                "Point at a Flask API, switch YOLO weights, and tune defaults. "
                "Changes apply to this NiceGUI app’s local engine; remote mode uses the API URL for detection."
            ).classes("so-sub")

            with ui.element("div").classes("so-panel").style("max-width: 720px"):
                ui.label("API").classes("so-panel-title")
                api_url = ui.input(
                    label="Backend API URL",
                    value=data.get("api_base_url", cfg.DEFAULTS["api_base_url"]),
                ).props("filled dense hide-bottom-space").classes("w-full")
                use_remote = ui.switch(
                    "Use remote API for detection",
                    value=bool(data.get("use_remote_api")),
                )
                ui.label(
                    "Off = run YOLO inside this process. On = POST images to the Flask /api/detect-object endpoint."
                ).classes("so-muted mb-2")
                ui.button(
                    "Test connection",
                    on_click=test_remote,
                ).props("outline no-caps")

            with ui.element("div").classes("so-panel").style("max-width: 720px"):
                ui.label("Local model").classes("so-panel-title")
                model_options = {m: m for m in cfg.KNOWN_MODELS}
                current_model = str(data.get("yolo_model") or cfg.DEFAULTS["yolo_model"])
                if current_model not in model_options:
                    model_options[current_model] = current_model
                model_select = ui.select(
                    model_options,
                    value=current_model if current_model in model_options else cfg.KNOWN_MODELS[0],
                    label="YOLO weights",
                ).props("filled dense hide-bottom-space").classes("w-full")
                custom_model = ui.input(
                    label="Or custom weights path / name",
                    value="" if current_model in cfg.KNOWN_MODELS else current_model,
                    placeholder="e.g. yolo11n.pt or C:\\models\\custom.pt",
                ).props("filled dense hide-bottom-space").classes("w-full")

                device_select = ui.select(
                    {"auto": "Auto", "cpu": "CPU", "0": "CUDA:0"},
                    value=str(data.get("device") or "auto"),
                    label="Device",
                ).props("filled dense hide-bottom-space").classes("w-full")

                ui.label("Default detection strength").classes(
                    "text-xs font-bold uppercase tracking-wide mt-2"
                ).style("color: var(--so-muted)")
                conf_slider = (
                    ui.slider(
                        min=0.1,
                        max=0.85,
                        step=0.05,
                        value=float(data.get("default_conf", 0.35)),
                    )
                    .props("label label-always color=cyan")
                    .classes("w-full")
                )
                imgsz_input = ui.number(
                    label="Inference image size",
                    value=int(data.get("imgsz", 640)),
                    min=320,
                    max=1280,
                    step=32,
                ).props("filled dense hide-bottom-space").classes("w-full")

                local_status = ui.label("").classes("so-muted mt-2")
                refresh_local_status()

            with ui.row().classes("gap-2 flex-wrap"):
                ui.button(
                    "Save & apply",
                    on_click=save_and_apply,
                ).props("color=primary unelevated no-caps")
                ui.button("Back to Organize", on_click=lambda: ui.navigate.to("/")).props(
                    "outline no-caps"
                )

            status_box["el"] = ui.label("").classes("so-muted mt-2")
