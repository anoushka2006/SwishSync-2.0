"""Basketball trajectory tracking from frame-level detections."""

from __future__ import annotations

from swishsync_cv.data import DetectionRecord, TrajectoryPoint


class BallTrajectoryTracker:
    """Track the basketball center by selecting one ball detection per frame.

    This intentionally starts simple: it records the highest-confidence
    basketball detection in each frame. That makes failures easy to inspect
    before adding more complex association logic later.
    """

    def __init__(self, min_confidence: float = 0.25) -> None:
        self.min_confidence = min_confidence
        self.points: list[TrajectoryPoint] = []

    def update(
        self,
        frame_index: int,
        timestamp_ms: float,
        detections: list[DetectionRecord],
    ) -> TrajectoryPoint | None:
        """Append the selected basketball center for this frame, if present."""

        basketball_detections = [
            detection
            for detection in detections
            if detection.label == "basketball"
            and detection.confidence >= self.min_confidence
        ]
        if not basketball_detections:
            return None

        selected = max(basketball_detections, key=lambda detection: detection.confidence)
        x, y = selected.center
        point = TrajectoryPoint(
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
            x=x,
            y=y,
            confidence=selected.confidence,
        )
        self.points.append(point)
        return point

    def reset(self) -> None:
        """Clear accumulated trajectory state."""

        self.points.clear()
