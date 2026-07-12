"""Place similar items together — no invented shelf/bin names."""

from __future__ import annotations

from typing import Any

# Soft synonym groups so "cup" and "wine glass" can cluster, etc.
SIMILARITY_GROUPS: list[set[str]] = [
    {"bottle", "cup", "wine glass", "wineglass"},
    {"book", "laptop", "keyboard", "mouse", "cell phone", "remote"},
    {"apple", "banana", "orange"},
    {"backpack", "handbag", "suitcase"},
    {"scissors", "knife", "fork", "spoon"},
]


def normalize_label(label: str) -> str:
    return " ".join(label.lower().strip().replace("_", " ").split())


def labels_similar(a: str, b: str) -> bool:
    a, b = normalize_label(a), normalize_label(b)
    if not a or not b:
        return False
    if a == b:
        return True
    if a in b or b in a:
        return True
    for group in SIMILARITY_GROUPS:
        if a in group and b in group:
            return True
    at, bt = set(a.split()), set(b.split())
    return bool(at & bt)


def _bbox_center(bbox: list[float]) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)


def _clamp_box(x1: float, y1: float, x2: float, y2: float) -> list[float]:
    x1, x2 = sorted((max(0.0, min(1.0, x1)), max(0.0, min(1.0, x2))))
    y1, y2 = sorted((max(0.0, min(1.0, y1)), max(0.0, min(1.0, y2))))
    if x2 - x1 < 0.06:
        cx = (x1 + x2) / 2
        x1, x2 = max(0.0, cx - 0.05), min(1.0, cx + 0.05)
    if y2 - y1 < 0.06:
        cy = (y1 + y2) / 2
        y1, y2 = max(0.0, cy - 0.05), min(1.0, cy + 0.05)
    return [round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4)]


def _overlaps(a: list[float], b: list[float], thresh: float = 0.15) -> bool:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return False
    inter = (ix2 - ix1) * (iy2 - iy1)
    area = max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))
    return inter / area >= thresh


def _candidate_near(similar: list[dict[str, Any]], occupied: list[list[float]]) -> list[float]:
    """Propose a box beside the similar cluster, avoiding overlaps."""
    xs = [_bbox_center(d["bbox"])[0] for d in similar]
    ys = [_bbox_center(d["bbox"])[1] for d in similar]
    cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
    # Typical item size from similars
    widths = [d["bbox"][2] - d["bbox"][0] for d in similar]
    heights = [d["bbox"][3] - d["bbox"][1] for d in similar]
    w = max(0.08, min(0.22, sum(widths) / len(widths)))
    h = max(0.08, min(0.22, sum(heights) / len(heights)))

    # Prefer right of rightmost similar, then left, below, above
    rightmost = max(similar, key=lambda d: d["bbox"][2])
    leftmost = min(similar, key=lambda d: d["bbox"][0])
    bottom = max(similar, key=lambda d: d["bbox"][3])
    top = min(similar, key=lambda d: d["bbox"][1])

    probes = [
        (rightmost["bbox"][2] + 0.02, cy - h / 2),
        (leftmost["bbox"][0] - w - 0.02, cy - h / 2),
        (cx - w / 2, bottom["bbox"][3] + 0.02),
        (cx - w / 2, top["bbox"][1] - h - 0.02),
        (cx + 0.08, cy - h / 2),
        (cx - 0.08 - w, cy - h / 2),
    ]
    for px, py in probes:
        box = _clamp_box(px, py, px + w, py + h)
        if not any(_overlaps(box, o) for o in occupied):
            return box
    # Fallback: slight offset from centroid even if slightly overlapping
    return _clamp_box(cx + 0.05, cy - h / 2, cx + 0.05 + w, cy + h / 2)


def _open_space(occupied: list[list[float]]) -> list[float]:
    """Pick the emptiest cell in a 3x4 grid (geometry only — no named bins)."""
    rows, cols = 3, 4
    best = None
    best_score = -1.0
    for r in range(rows):
        for c in range(cols):
            cell = [
                c / cols + 0.04 / cols,
                r / rows + 0.04 / rows,
                (c + 1) / cols - 0.04 / cols,
                (r + 1) / rows - 0.04 / rows,
            ]
            hit = sum(1 for o in occupied if _overlaps(cell, o, thresh=0.12))
            # Prefer center-ish empty cells
            cx, cy = _bbox_center(cell)
            center_bonus = 1.0 - ((cx - 0.5) ** 2 + (cy - 0.5) ** 2) ** 0.5
            score = (10 - hit) + center_bonus
            if score > best_score:
                best_score = score
                best = cell
    return best or [0.4, 0.4, 0.6, 0.6]


def suggest_similar_placement(
    item_label: str,
    detections: list[dict[str, Any]],
    *,
    min_conf: float = 0.25,
) -> dict[str, Any]:
    """
    Find where to put an item by grouping with similar detections in-frame.
    Returns anchor_box + human recommendation — never invented shelf names.
    """
    occupied = [
        d["bbox"]
        for d in detections
        if d.get("bbox") and float(d.get("confidence", 0)) >= min_conf
    ]
    similar = [
        d
        for d in detections
        if d.get("bbox")
        and float(d.get("confidence", 0)) >= min_conf
        and labels_similar(item_label, str(d.get("label", "")))
    ]

    if similar:
        anchor = _candidate_near(similar, occupied)
        peer = normalize_label(str(similar[0].get("label", item_label)))
        count = len(similar)
        target = f"With similar items ({peer})"
        recommendation = (
            f'Place near the {count} similar "{peer}" item{"s" if count != 1 else ""} '
            f"already in view."
        )
        mode = "similar_cluster"
    else:
        anchor = _open_space(occupied)
        target = "Open space"
        recommendation = (
            f'No similar "{normalize_label(item_label)}" items in view — '
            f"use this open area."
        )
        mode = "open_space"

    cx, cy = _bbox_center(anchor)
    return {
        "target": target,
        "recommendation": recommendation,
        "anchor_box": anchor,
        "mode": mode,
        "similar_count": len(similar),
        "guidance": {
            "arrows": ["HOLD"] if abs(cx - 0.5) < 0.12 and abs(cy - 0.5) < 0.12 else (
                (["LEFT"] if cx < 0.5 else ["RIGHT"])
                + (["UP"] if cy < 0.5 else ["DOWN"])
            ),
            "aligned": abs(cx - 0.5) < 0.12 and abs(cy - 0.5) < 0.12,
        },
        "coordinates": {"x": round(cx * 4 - 2, 2), "y": round((1 - cy) * 2, 2), "z": -1.0},
        "rotation": 0,
    }
