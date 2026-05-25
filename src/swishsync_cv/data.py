"""Shared data contracts used across the CV pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

DetectionCategory = Literal["basketball", "hoop"]
ShotLifecycleState = Literal["idle", "collecting_shot", "shot_finalized"]


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


@dataclass(frozen=True)
class SparseBallDetection:
    """A sparse basketball detection stored for shot reconstruction."""

    frame_index: int
    timestamp_ms: float
    x: float
    y: float
    confidence: float
    interpolated: bool = False


@dataclass(frozen=True)
class HoopLock:
    """Locked hoop position used as a stable anchor across frames."""

    center_x: float
    center_y: float
    bbox_xyxy: tuple[float, float, float, float]
    confidence: float
    locked_at_frame: int
    is_locked: bool = False
    detector_confidence: float = 0.0
    color_score: float = 0.0


@dataclass(frozen=True)
class ParabolaFit:
    """Quadratic fit of shot points: y = a*x^2 + b*x + c."""

    coefficients: tuple[float, float, float]
    r_squared: float
    apex_x: float
    apex_y: float
    x_min: float
    x_max: float
    weighted_r_squared: float = 0.0
    weighted_residual_rmse: float = 0.0

    def evaluate_y(self, x: float) -> float:
        a, b, c = self.coefficients
        return a * x * x + b * x + c

    def sample_arc(self, num_points: int = 96) -> list[tuple[float, float]]:
        if num_points < 2:
            return []
        xs = np.linspace(self.x_min, self.x_max, num_points)
        return [(float(x), float(self.evaluate_y(x))) for x in xs]


@dataclass(frozen=True)
class PointDiagnostic:
    """Per-point fit diagnostic used for debugging trajectory quality."""

    frame_index: int
    x: float
    y: float
    confidence: float
    fitting_weight: float
    residual_px: float
    is_outlier: bool


@dataclass(frozen=True)
class FitDiagnostics:
    """Aggregate diagnostics for a confidence-weighted parabola fit."""

    point_count: int
    average_detection_confidence: float
    weighted_residual_rmse: float
    outlier_count: int
    points: tuple[PointDiagnostic, ...]


@dataclass
class ConfidenceScores:
    """Composite confidence metrics for a shot candidate."""

    detection_confidence: float
    trajectory_confidence: float
    overall_confidence: float

    def as_percentages(self) -> tuple[int, int, int]:
        return (
            int(round(self.detection_confidence * 100)),
            int(round(self.trajectory_confidence * 100)),
            int(round(self.overall_confidence * 100)),
        )


@dataclass
class ShotCandidate:
    """Buffered shot attempt collected before single-pass reconstruction."""

    start_frame: int
    end_frame: int | None = None
    candidate_points: list[SparseBallDetection] = field(default_factory=list)
    validated_points: list[SparseBallDetection] = field(default_factory=list)
    parabola_fit: ParabolaFit | None = None
    fit_diagnostics: FitDiagnostics | None = None
    confidence: ConfidenceScores | None = None
    state: ShotLifecycleState = "collecting_shot"
    post_rim_frames_remaining: int = 0

    @property
    def raw_points(self) -> list[SparseBallDetection]:
        """Backward-compatible alias for buffered collection points."""

        return self.candidate_points


@dataclass(frozen=True)
class ShotAxis:
    """Canonical shooter-to-hoop axis used for normalized shot-space projection."""

    shooter_x: float
    shooter_y: float
    hoop_x: float
    hoop_y: float
    axis_dx: float
    axis_dy: float
    scale: float

    @property
    def unit_along(self) -> tuple[float, float]:
        length = (self.axis_dx**2 + self.axis_dy**2) ** 0.5
        if length < 1e-6:
            return (1.0, 0.0)
        return (self.axis_dx / length, self.axis_dy / length)

    @property
    def unit_perp(self) -> tuple[float, float]:
        along_x, along_y = self.unit_along
        return (-along_y, along_x)


@dataclass(frozen=True)
class CompletedShot:
    """Persistent finalized shot stored in analytical shot memory."""

    shot_id: int
    start_frame: int
    end_frame: int
    timestamp_ms: float
    arc_points: list[tuple[float, float]]
    normalized_arc_points: list[tuple[float, float]]
    normalized_apex: tuple[float, float]
    shot_axis: ShotAxis
    confidence: ConfidenceScores
    parabola_r_squared: float
    outcome: Literal["unknown"] = "unknown"
