"""Shared chrome + styles for NiceGUI pages."""

from __future__ import annotations

from nicegui import ui

PACKAGE_SVG = """
<svg class="so-logo" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
  <rect x="6" y="14" width="36" height="28" rx="8" fill="url(#g)" opacity="0.2"/>
  <path d="M10 18.5L24 10l14 8.5V36a6 6 0 01-6 6H16a6 6 0 01-6-6V18.5z" stroke="url(#g)" stroke-width="2.4" stroke-linejoin="round"/>
  <path d="M10 18.5L24 27l14-8.5" stroke="url(#g)" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/>
  <path d="M24 27v15" stroke="url(#g)" stroke-width="2.4" stroke-linecap="round"/>
  <defs>
    <linearGradient id="g" x1="8" y1="8" x2="40" y2="42" gradientUnits="userSpaceOnUse">
      <stop stop-color="#22d3ee"/><stop offset="1" stop-color="#3b82f6"/>
    </linearGradient>
  </defs>
</svg>
"""

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

:root {
  --so-radius: 1rem;
  --so-radius-sm: 0.75rem;
  --so-control: 0.75rem;
  --so-ink: #0f172a;
  --so-muted: #64748b;
  --so-surface: rgba(255, 255, 255, 0.88);
  --so-border: rgba(14, 165, 233, 0.18);
  --so-accent: #0891b2;
  --so-accent-2: #22d3ee;
  --so-accent-soft: rgba(8, 145, 178, 0.12);
  --so-field-bg: rgba(8, 145, 178, 0.06);
  --so-glow:
    radial-gradient(1100px 560px at 8% -12%, rgba(34, 211, 238, 0.22), transparent 55%),
    radial-gradient(900px 500px at 96% 0%, rgba(59, 130, 246, 0.16), transparent 50%),
    linear-gradient(180deg, #f0f9ff 0%, #e0f2fe 42%, #f8fafc 100%);
}
body.body--dark {
  --so-ink: #e2e8f0;
  --so-muted: #94a3b8;
  --so-surface: rgba(15, 23, 42, 0.82);
  --so-border: rgba(34, 211, 238, 0.18);
  --so-accent: #22d3ee;
  --so-accent-2: #67e8f9;
  --so-accent-soft: rgba(34, 211, 238, 0.14);
  --so-field-bg: rgba(34, 211, 238, 0.08);
  --so-glow:
    radial-gradient(1000px 520px at 10% -10%, rgba(34, 211, 238, 0.14), transparent 55%),
    radial-gradient(800px 480px at 95% 5%, rgba(59, 130, 246, 0.12), transparent 50%),
    linear-gradient(180deg, #0b1220 0%, #0f172a 100%);
}
html, body, #app { min-height: 100%; height: 100%; }
body {
  font-family: 'Plus Jakarta Sans', system-ui, sans-serif !important;
  background: var(--so-glow) !important;
  color: var(--so-ink) !important;
  margin: 0 !important;
}
.q-page, .nicegui-content { padding: 0 !important; max-width: none !important; }
.so-shell { min-height: 100vh; width: 100%; display: flex; flex-direction: column; }
.so-header {
  position: sticky; top: 0; z-index: 40;
  display: flex; align-items: center; justify-content: space-between; gap: 0.75rem;
  padding: 0.85rem 1.25rem;
  background: color-mix(in srgb, var(--so-surface) 88%, transparent);
  backdrop-filter: blur(14px);
  border-bottom: 1px solid var(--so-border);
}
.so-brand-wrap { display: flex; align-items: center; gap: 0.7rem; animation: so-fade-up 0.55s ease both; }
.so-logo { width: 2.35rem; height: 2.35rem; flex-shrink: 0; }
.so-brand {
  font-weight: 800; letter-spacing: -0.03em; font-size: 1.35rem; line-height: 1.1;
  background: linear-gradient(120deg, var(--so-ink) 20%, var(--so-accent));
  -webkit-background-clip: text; background-clip: text; color: transparent;
}
.so-main {
  flex: 1; width: 100%; max-width: 1400px; margin: 0 auto;
  padding: 1.25rem 1.25rem 2.5rem; display: flex; flex-direction: column; gap: 1.1rem;
}
.so-sub { color: var(--so-muted); font-size: 0.98rem; max-width: 42rem; animation: so-fade-up 0.6s ease 0.05s both; }
.so-steps { display: flex; gap: 0.35rem; overflow-x: auto; animation: so-fade-up 0.55s ease 0.08s both; }
.so-step {
  display: inline-flex; align-items: center; gap: 0.45rem; white-space: nowrap;
  padding: 0.4rem 0.7rem; border-radius: 999px; font-size: 0.82rem; font-weight: 600;
  color: var(--so-muted); border: 1px solid transparent;
  transition: background 0.25s ease, color 0.25s ease, border-color 0.25s ease;
}
.so-step.is-current {
  color: var(--so-accent); background: var(--so-accent-soft);
  border-color: color-mix(in srgb, var(--so-accent) 25%, transparent);
}
.so-step.is-done { color: var(--so-ink); }
.so-step-num {
  width: 1.35rem; height: 1.35rem; border-radius: 999px;
  display: inline-flex; align-items: center; justify-content: center;
  font-size: 0.72rem; background: var(--so-accent-soft); color: var(--so-accent);
}
.so-step.is-current .so-step-num { background: var(--so-accent); color: white; }
.so-grid {
  display: grid; grid-template-columns: minmax(280px, 360px) minmax(0, 1fr);
  gap: 1.1rem; align-items: start; flex: 1;
}
@media (max-width: 900px) {
  .so-grid { grid-template-columns: 1fr; }
  .so-main { padding: 1rem 0.85rem 2rem; }
  .so-brand { font-size: 1.15rem; }
}
.so-panel {
  background: var(--so-surface); border: 1px solid var(--so-border);
  backdrop-filter: blur(14px); border-radius: var(--so-radius);
  box-shadow: 0 16px 40px rgba(8, 145, 178, 0.06);
  padding: 1.1rem; animation: so-fade-up 0.55s ease both;
}
.so-panel-title { font-size: 1.05rem; font-weight: 700; margin-bottom: 0.75rem; letter-spacing: -0.02em; }
.so-frame {
  position: relative; overflow: hidden; border-radius: var(--so-radius);
  border: 1px solid var(--so-border); background: #020617;
  min-height: min(58vh, 560px); aspect-ratio: 4 / 3; width: 100%;
}
.so-frame video, .so-frame img, .so-frame .nicegui-interactive-image {
  width: 100% !important; height: 100% !important; object-fit: contain; display: block;
}
.so-frame video {
  /* native video — never replace src with JPEG → no pulse */
  background: #020617;
}
.so-overlay-layer {
  position: absolute;
  inset: 0;
  pointer-events: none;
  z-index: 6;
  overflow: hidden;
}
.so-live-box {
  position: absolute;
  border: 2px solid #22d3ee;
  border-radius: 12px;
  box-sizing: border-box;
  transition: left 0.15s ease, top 0.15s ease, width 0.15s ease, height 0.15s ease;
}
.so-live-box.is-anchor {
  border-style: dashed;
  border-width: 2px;
}
.so-live-tag {
  position: absolute;
  left: 0;
  top: -1.35rem;
  max-width: 100%;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  padding: 0.15rem 0.45rem 0.15rem 0.55rem;
  border-radius: 0.4rem;
  background: rgba(15, 23, 42, 0.88);
  color: #f8fafc;
  font-size: 0.7rem;
  font-weight: 700;
  border-left: 3px solid #22d3ee;
}
.so-frame.is-empty::before {
  content: 'Preview'; position: absolute; inset: 0; display: flex;
  align-items: center; justify-content: center; color: #64748b; font-weight: 600;
  pointer-events: none; z-index: 1;
}
.so-pill {
  display: inline-flex; align-items: center; gap: 0.35rem;
  padding: 0.35rem 0.75rem; border-radius: 999px;
  background: var(--so-accent-soft); color: var(--so-accent);
  font-size: 0.78rem; font-weight: 700;
}
.so-item {
  display: flex; align-items: center; gap: 0.75rem; width: 100%; text-align: left;
  padding: 0.7rem 0.85rem; border-radius: var(--so-radius-sm);
  border: 1px solid var(--so-border); background: transparent; color: var(--so-ink);
  font-weight: 600; cursor: pointer;
  transition: border-color 0.2s ease, transform 0.2s ease, background 0.2s ease;
}
.so-item:hover { border-color: color-mix(in srgb, var(--so-accent) 45%, transparent); transform: translateY(-1px); }
.so-item.is-selected {
  border-color: var(--so-accent); background: var(--so-accent-soft);
}
.so-item-dot { width: 0.65rem; height: 0.65rem; border-radius: 999px; background: var(--so-accent-2); flex-shrink: 0; }
.so-result { animation: so-pop 0.4s cubic-bezier(0.22, 1, 0.36, 1) both; }
.so-muted { color: var(--so-muted); font-size: 0.85rem; }
.so-error { color: #e11d48; font-size: 0.85rem; font-weight: 600; }

/* Shared control radius — buttons, fields, upload */
.q-btn {
  font-weight: 700 !important;
  text-transform: none !important;
  border-radius: var(--so-control) !important;
  min-height: 2.65rem !important;
}

/* Kill Quasar outlined "notch / tail" borders — use a clean rounded box */
.so-panel .q-field {
  margin-top: 0.15rem;
}
.so-panel .q-field__control,
.so-panel .q-field--outlined .q-field__control,
.so-panel .q-field--filled .q-field__control,
.so-panel .q-field--standard .q-field__control {
  border-radius: var(--so-control) !important;
  background: var(--so-field-bg) !important;
  min-height: 2.65rem !important;
}
.so-panel .q-field__control:before,
.so-panel .q-field__control:after {
  border: none !important;
  border-radius: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
  height: auto !important;
}
.so-panel .q-field__control {
  border: 1px solid var(--so-border) !important;
  padding: 0 0.85rem !important;
}
.so-panel .q-field--focused .q-field__control {
  border-color: var(--so-accent) !important;
  box-shadow: 0 0 0 1px color-mix(in srgb, var(--so-accent) 35%, transparent);
}
.so-panel .q-field__bottom {
  padding-top: 0.25rem;
}
.so-panel .q-field__label {
  color: var(--so-muted) !important;
}
.so-panel .q-field__native,
.so-panel .q-field__input,
.so-panel .q-field__prefix,
.so-panel .q-field__suffix {
  color: var(--so-ink) !important;
}

/* Upload — same radius on every corner */
.so-panel .q-uploader,
.so-panel .so-upload,
.so-panel .so-upload .q-uploader {
  border-radius: var(--so-control) !important;
  border: 1px solid var(--so-border) !important;
  overflow: hidden !important;
  background: var(--so-field-bg) !important;
  box-shadow: none !important;
}
.so-panel .q-uploader__header {
  border-radius: 0 !important;
  background: color-mix(in srgb, var(--so-accent) 18%, transparent) !important;
}
.so-panel .q-uploader__list {
  border-radius: 0 !important;
  min-height: 4.5rem;
  background: transparent !important;
}
.so-panel .q-uploader .q-btn {
  border-radius: calc(var(--so-control) - 2px) !important;
  min-height: 2rem !important;
}

/* Filled fields: kill underline + leftover notch pieces */
.so-panel .q-field--filled .q-field__control:before,
.so-panel .q-field--standard .q-field__control:before {
  display: none !important;
}
.so-panel .q-field__marginal {
  height: auto !important;
}

/* Dropdown menu matches controls */
.q-menu {
  border-radius: var(--so-control) !important;
  border: 1px solid var(--so-border);
  overflow: hidden;
}

@keyframes so-fade-up {
  from { opacity: 0; transform: translateY(10px); }
  to { opacity: 1; transform: translateY(0); }
}
@keyframes so-pop {
  from { opacity: 0; transform: scale(0.97) translateY(6px); }
  to { opacity: 1; transform: scale(1) translateY(0); }
}
"""

CUSTOM_CSS += """
.so-nav {
  display: inline-flex; align-items: center;
  padding: 0.4rem 0.85rem; border-radius: var(--so-control);
  font-size: 0.85rem; font-weight: 700; text-decoration: none !important;
  color: var(--so-muted); border: 1px solid transparent;
  transition: background 0.2s ease, color 0.2s ease, border-color 0.2s ease;
}
.so-nav:hover { color: var(--so-ink); background: var(--so-accent-soft); }
.so-nav.is-active {
  color: var(--so-accent); background: var(--so-accent-soft);
  border-color: color-mix(in srgb, var(--so-accent) 28%, transparent);
}
"""


def inject_styles() -> None:
    ui.add_head_html(f"<style>{CUSTOM_CSS}</style>")


def render_header(*, active: str = "organize") -> None:
    dark = ui.dark_mode(value=None)
    with ui.element("div").classes("so-header"):
        with ui.element("div").classes("so-brand-wrap"):
            ui.html(PACKAGE_SVG)
            ui.label("Smart Organizer").classes("so-brand")

        with ui.row().classes("items-center gap-2 flex-wrap"):
            ui.link("Organize", "/").classes(
                "so-nav is-active" if active == "organize" else "so-nav"
            )
            ui.link("Backend", "/backend").classes(
                "so-nav is-active" if active == "backend" else "so-nav"
            )

            theme_cycle = {"value": 0}

            def cycle_theme() -> None:
                theme_cycle["value"] = (theme_cycle["value"] + 1) % 3
                if theme_cycle["value"] == 0:
                    dark.auto()
                    theme_btn.set_text("Auto")
                elif theme_cycle["value"] == 1:
                    dark.disable()
                    theme_btn.set_text("Light")
                else:
                    dark.enable()
                    theme_btn.set_text("Dark")

            theme_btn = ui.button("Auto", on_click=cycle_theme).props("flat dense no-caps")
