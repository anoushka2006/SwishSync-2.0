"""Sparse basketball detection scheduling and interpolation."""

from __future__ import annotations

from swishsync_cv.config import SparseDetectionConfig
from swishsync_cv.data import DetectionRecord, SparseBallDetection


class SparseBallDetectionBuffer:
    """Collect sparse ball detections and infer positions between frames."""

    def __init__(self, config: SparseDetectionConfig) -> None:
        self.config = config
        self.detections: list[SparseBallDetection] = []

    def should_detect(
        self,
        frame_index: int,
        *,
        dense: bool = False,
        idle: bool = False,
    ) -> bool:
        if dense or idle:
            return True
        stride = max(1, self.config.detection_stride)
        return frame_index % stride == 0

    def extract_ball_detection(
        self,
        frame_index: int,
        timestamp_ms: float,
        detections: list[DetectionRecord],
    ) -> SparseBallDetection | None:
        basketball_detections = [
            detection
            for detection in detections
            if detection.label == "basketball"
            and detection.confidence >= self.config.min_confidence
        ]
        if not basketball_detections:
            return None

        selected = max(basketball_detections, key=lambda detection: detection.confidence)
        x, y = selected.center
        point = SparseBallDetection(
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
            x=x,
            y=y,
            confidence=selected.confidence,
            interpolated=False,
        )
        self.detections.append(point)
        return point

    def register_detection(self, point: SparseBallDetection) -> None:
        """Record an externally recovered measured detection."""

        self.detections.append(point)

    def interpolate_at(self, frame_index: int, timestamp_ms: float) -> SparseBallDetection | None:
        """Linearly infer ball position between the two nearest sparse detections."""

        if len(self.detections) < 2:
            return self.detections[-1] if self.detections else None

        previous = self.detections[-2]
        latest = self.detections[-1]
        if frame_index <= latest.frame_index:
            return latest

        frame_gap = latest.frame_index - previous.frame_index
        if frame_gap <= 0:
            return latest

        alpha = (frame_index - latest.frame_index) / frame_gap
        alpha = min(alpha, 1.0)
        return SparseBallDetection(
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
            x=latest.x + alpha * (latest.x - previous.x),
            y=latest.y + alpha * (latest.y - previous.y),
            confidence=latest.confidence * 0.85,
            interpolated=True,
        )

    def latest(self) -> SparseBallDetection | None:
        return self.detections[-1] if self.detections else None

    def recent(self, count: int) -> list[SparseBallDetection]:
        return self.detections[-count:]
