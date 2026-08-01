"""Hoop bbox parsing and rim/hoop center helpers."""

from __future__ import annotations


def parse_hoop_bbox_arg(value: str) -> tuple[float, float, float, float]:
    """Parse ``x,y,w,h`` manual hoop bbox from CLI input."""

    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 4:
        raise ValueError("Hoop bbox must be four comma-separated values: x,y,w,h")
    x, y, width, height = (float(part) for part in parts)
    if width <= 0 or height <= 0:
        raise ValueError("Hoop bbox width and height must be positive.")
    return x, y, width, height


def bbox_xywh_to_xyxy(x: float, y: float, width: float, height: float) -> tuple[float, float, float, float]:
    return x, y, x + width, y + height


def hoop_center_from_bbox(bbox_xyxy: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox_xyxy
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def rim_center_from_bbox(bbox_xyxy: tuple[float, float, float, float]) -> tuple[float, float]:
    """Return the bottom-center of the hoop bbox as the rim anchor."""

    x1, _y1, x2, y2 = bbox_xyxy
    return ((x1 + x2) / 2.0, y2)
