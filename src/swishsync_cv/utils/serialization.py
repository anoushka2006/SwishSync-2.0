"""Serialization helpers for detection and trajectory debugging artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from swishsync_cv.data import DetectionRecord, FrameDetections, TrajectoryPoint


def detection_to_dict(detection: DetectionRecord) -> dict[str, object]:
    x1, y1, x2, y2 = detection.bbox_xyxy
    center_x, center_y = detection.center
    return {
        "frame_index": detection.frame_index,
        "timestamp_ms": detection.timestamp_ms,
        "label": detection.label,
        "class_name": detection.class_name,
        "confidence": detection.confidence,
        "bbox_x1": x1,
        "bbox_y1": y1,
        "bbox_x2": x2,
        "bbox_y2": y2,
        "center_x": center_x,
        "center_y": center_y,
    }


def trajectory_point_to_dict(point: TrajectoryPoint) -> dict[str, object]:
    return {
        "frame_index": point.frame_index,
        "timestamp_ms": point.timestamp_ms,
        "x": point.x,
        "y": point.y,
        "confidence": point.confidence,
    }


def write_detections_jsonl(
    path: Path,
    frame_detections: list[FrameDetections],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for frame_record in frame_detections:
            payload = {
                "frame_index": frame_record.frame_index,
                "timestamp_ms": frame_record.timestamp_ms,
                "detections": [
                    detection_to_dict(detection)
                    for detection in frame_record.detections
                ],
            }
            file.write(json.dumps(payload) + "\n")


def write_detections_csv(path: Path, detections: list[DetectionRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "frame_index",
        "timestamp_ms",
        "label",
        "class_name",
        "confidence",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "center_x",
        "center_y",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for detection in detections:
            writer.writerow(detection_to_dict(detection))


def write_trajectory_json(path: Path, trajectory: list[TrajectoryPoint]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [trajectory_point_to_dict(point) for point in trajectory]
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
