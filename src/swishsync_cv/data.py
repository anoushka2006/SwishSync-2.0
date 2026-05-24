"""Shared data contracts used across the CV pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

DetectionCategory = Literal["basketball", "hoop"]


@dataclass(frozen=True)
class VideoMetadata:
    """OpenCV video metadata normalized for downstream modules."""

    path: Path
    fps: float
    width: int
    height: int
    frame_count: int

    @property
    def frame_size(self) -> tuple[int, int]:
        """Return frame size in the OpenCV writer order: (width, height)."""

        return (self.width, self.height)


@dataclass(frozen=True)
class FramePacket:
    """A single decoded video frame plus stable frame timing metadata."""

    index: int
    timestamp_ms: float
    image: np.ndarray


@dataclass(frozen=True)
class DetectionRecord:
    """Detector output for one object in one frame."""

    frame_index: int
    timestamp_ms: float
    label: DetectionCategory
    class_name: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]

    @property
    def center(self) -> tuple[float, float]:
        """Return bbox center as (x, y) pixel coordinates."""

        x1, y1, x2, y2 = self.bbox_xyxy
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


@dataclass(frozen=True)
class TrajectoryPoint:
    """Tracked basketball center for one frame."""

    frame_index: int
    timestamp_ms: float
    x: float
    y: float
    confidence: float


@dataclass(frozen=True)
class FrameDetections:
    """All detections produced for one video frame."""

    frame_index: int
    timestamp_ms: float
    detections: list[DetectionRecord]
