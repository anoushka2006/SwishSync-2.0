"""Hybrid hoop detection using YOLO, orange color filtering, and geometry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

from swishsync_cv.config import HoopLockConfig
from swishsync_cv.data import DetectionRecord

HoopCandidateSource = Literal["yolo", "color", "hybrid"]


@dataclass(frozen=True)
class HoopCandidate:
    """One scored hoop hypothesis in camera space."""

    bbox_xyxy: tuple[float, float, float, float]
    center_x: float
    center_y: float
    detector_confidence: float
    color_score: float
    geometry_score: float
    composite_score: float
    source: HoopCandidateSource


class HybridHoopDetector:
    """Detect hoop candidates with detector, color, and shape cues."""

    def __init__(self, config: HoopLockConfig) -> None:
        self.config = config

    def detect(
        self,
        frame: np.ndarray,
        yolo_hoop_detections: list[DetectionRecord],
    ) -> list[HoopCandidate]:
        candidates: list[HoopCandidate] = []
        frame_height, frame_width = frame.shape[:2]

        for detection in yolo_hoop_detections:
            candidate = self._score_bbox(
                frame=frame,
                bbox_xyxy=detection.bbox_xyxy,
                detector_confidence=detection.confidence,
                source="yolo",
                frame_width=frame_width,
                frame_height=frame_height,
            )
            if candidate is not None:
                candidates.append(candidate)

        for bbox in self._find_orange_regions(frame):
            if _overlaps_existing(candidates, bbox):
                continue
            candidate = self._score_bbox(
                frame=frame,
                bbox_xyxy=bbox,
                detector_confidence=0.0,
                source="color",
                frame_width=frame_width,
                frame_height=frame_height,
            )
            if candidate is not None:
                candidates.append(candidate)

        for index, yolo_candidate in enumerate(list(candidates)):
            if yolo_candidate.source != "yolo":
                continue
            for color_candidate in candidates:
                if color_candidate.source != "color":
                    continue
                if _bbox_iou(yolo_candidate.bbox_xyxy, color_candidate.bbox_xyxy) < 0.15:
                    continue
                candidates[index] = _merge_candidates(yolo_candidate, color_candidate)

        candidates.sort(key=lambda candidate: candidate.composite_score, reverse=True)
        return candidates

    def _score_bbox(
        self,
        frame: np.ndarray,
        bbox_xyxy: tuple[float, float, float, float],
        detector_confidence: float,
        source: HoopCandidateSource,
        frame_width: int,
        frame_height: int,
    ) -> HoopCandidate | None:
        x1, y1, x2, y2 = bbox_xyxy
        width = max(x2 - x1, 1.0)
        height = max(y2 - y1, 1.0)
        geometry_score = _geometry_score(
            width=width,
            height=height,
            center_y=(y1 + y2) / 2.0,
            frame_width=frame_width,
            frame_height=frame_height,
            config=self.config,
        )
        if geometry_score <= 0.0:
            return None

        color_score = _color_score(frame, bbox_xyxy, self.config)
        if source == "yolo" and color_score >= 0.20:
            source = "hybrid"
        composite_score = _composite_score(
            detector_confidence=detector_confidence,
            color_score=color_score,
            geometry_score=geometry_score,
            source=source,
        )
        if composite_score < 0.12:
            return None

        return HoopCandidate(
            bbox_xyxy=bbox_xyxy,
            center_x=(x1 + x2) / 2.0,
            center_y=(y1 + y2) / 2.0,
            detector_confidence=detector_confidence,
            color_score=color_score,
            geometry_score=geometry_score,
            composite_score=composite_score,
            source=source,
        )

    def _find_orange_regions(self, frame: np.ndarray) -> list[tuple[float, float, float, float]]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.array(
            [
                self.config.orange_hue_min,
                self.config.orange_sat_min,
                self.config.orange_val_min,
            ],
            dtype=np.uint8,
        )
        upper = np.array(
            [
                self.config.orange_hue_max,
                self.config.orange_sat_max,
                self.config.orange_val_max,
            ],
            dtype=np.uint8,
        )
        mask = cv2.inRange(hsv, lower, upper)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes: list[tuple[float, float, float, float]] = []
        frame_height, frame_width = frame.shape[:2]
        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)
            bbox = (float(x), float(y), float(x + width), float(y + height))
            if _geometry_score(
                width=float(width),
                height=float(height),
                center_y=float(y + height / 2),
                frame_width=frame_width,
                frame_height=frame_height,
                config=self.config,
            ) <= 0.0:
                continue
            boxes.append(bbox)
        return boxes


def _geometry_score(
    width: float,
    height: float,
    center_y: float,
    frame_width: int,
    frame_height: int,
    config: HoopLockConfig,
) -> float:
    if width < config.min_hoop_width_px or height < config.min_hoop_height_px:
        return 0.0
    if width > frame_width * config.max_hoop_width_ratio:
        return 0.0
    if height > frame_height * config.max_hoop_height_ratio:
        return 0.0
    if center_y > frame_height * config.upper_frame_ratio:
        return 0.0

    aspect_ratio = width / max(height, 1.0)
    if aspect_ratio < config.min_aspect_ratio or aspect_ratio > config.max_aspect_ratio:
        return 0.0

    size_score = min(width / max(frame_width * 0.05, 1.0), 1.0)
    position_score = 1.0 - (center_y / max(frame_height * config.upper_frame_ratio, 1.0))
    aspect_score = 1.0 - abs(aspect_ratio - 2.0) / 2.5
    return max(0.0, min(1.0, 0.45 * size_score + 0.30 * position_score + 0.25 * aspect_score))


def _color_score(
    frame: np.ndarray,
    bbox_xyxy: tuple[float, float, float, float],
    config: HoopLockConfig,
) -> float:
    x1, y1, x2, y2 = [int(round(value)) for value in bbox_xyxy]
    height, width = frame.shape[:2]
    x1 = max(0, min(x1, width - 1))
    x2 = max(x1 + 1, min(x2, width))
    y1 = max(0, min(y1, height - 1))
    y2 = max(y1 + 1, min(y2, height))

    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower = np.array(
        [config.orange_hue_min, config.orange_sat_min, config.orange_val_min],
        dtype=np.uint8,
    )
    upper = np.array(
        [config.orange_hue_max, config.orange_sat_max, config.orange_val_max],
        dtype=np.uint8,
    )
    mask = cv2.inRange(hsv, lower, upper)
    return float(np.count_nonzero(mask) / max(mask.size, 1))


def _composite_score(
    detector_confidence: float,
    color_score: float,
    geometry_score: float,
    source: HoopCandidateSource,
) -> float:
    if source == "hybrid":
        detector_weight = 0.35
    elif source == "yolo":
        detector_weight = 0.40
    else:
        detector_weight = 0.10

    score = (
        detector_weight * detector_confidence
        + 0.30 * color_score
        + 0.30 * geometry_score
    )
    if source == "hybrid":
        score += 0.10
    return max(0.0, min(1.0, score))


def _bbox_iou(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if intersection <= 0.0:
        return 0.0
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


def _overlaps_existing(
    candidates: list[HoopCandidate],
    bbox: tuple[float, float, float, float],
) -> bool:
    return any(_bbox_iou(candidate.bbox_xyxy, bbox) >= 0.35 for candidate in candidates)


def _merge_candidates(first: HoopCandidate, second: HoopCandidate) -> HoopCandidate:
    x1 = min(first.bbox_xyxy[0], second.bbox_xyxy[0])
    y1 = min(first.bbox_xyxy[1], second.bbox_xyxy[1])
    x2 = max(first.bbox_xyxy[2], second.bbox_xyxy[2])
    y2 = max(first.bbox_xyxy[3], second.bbox_xyxy[3])
    detector_confidence = max(first.detector_confidence, second.detector_confidence)
    color_score = max(first.color_score, second.color_score)
    geometry_score = max(first.geometry_score, second.geometry_score)
    composite_score = _composite_score(
        detector_confidence=detector_confidence,
        color_score=color_score,
        geometry_score=geometry_score,
        source="hybrid",
    )
    return HoopCandidate(
        bbox_xyxy=(x1, y1, x2, y2),
        center_x=(x1 + x2) / 2.0,
        center_y=(y1 + y2) / 2.0,
        detector_confidence=detector_confidence,
        color_score=color_score,
        geometry_score=geometry_score,
        composite_score=composite_score,
        source="hybrid",
    )
