"""Pre-sparse filters for full-frame basketball detections."""

from __future__ import annotations

from swishsync_cv.data import DetectionRecord, HoopLock, SparseBallDetection
from swishsync_cv.tracking.parabola import is_floor_bounce_point


def filter_basketball_detections(
    detections: list[DetectionRecord],
    hoop_lock: HoopLock | None,
    *,
    floor_margin_px: float,
    enabled: bool = True,
) -> list[DetectionRecord]:
    """Drop hoop-locked floor-band basketball labels before sparse extraction."""

    if not enabled or hoop_lock is None or not hoop_lock.is_locked:
        return detections

    filtered: list[DetectionRecord] = []
    for detection in detections:
        if detection.label != "basketball":
            filtered.append(detection)
            continue
        if _is_floor_band_detection(detection, hoop_lock, floor_margin_px):
            continue
        filtered.append(detection)
    return filtered


def _is_floor_band_detection(
    detection: DetectionRecord,
    hoop_lock: HoopLock,
    floor_margin_px: float,
) -> bool:
    center_x, center_y = detection.center
    point = SparseBallDetection(
        frame_index=detection.frame_index,
        timestamp_ms=detection.timestamp_ms,
        x=center_x,
        y=center_y,
        confidence=detection.confidence,
    )
    return is_floor_bounce_point(
        point,
        hoop_lock,
        floor_margin_px=floor_margin_px,
    )
