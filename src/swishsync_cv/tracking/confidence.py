"""Composite confidence scoring for reconstructed shots."""

from __future__ import annotations

from swishsync_cv.data import ConfidenceScores, HoopLock, ParabolaFit, ShotCandidate, SparseBallDetection


def score_shot_confidence(
    candidate: ShotCandidate,
    hoop_lock: HoopLock | None = None,
) -> ConfidenceScores:
    """Compute detection, trajectory, and overall confidence for a shot."""

    detection_confidence = _detection_confidence(candidate.validated_points)
    trajectory_confidence = _trajectory_confidence(
        validated_points=candidate.validated_points,
        parabola_fit=candidate.parabola_fit,
        hoop_lock=hoop_lock,
    )
    overall_confidence = 0.45 * detection_confidence + 0.55 * trajectory_confidence
    return ConfidenceScores(
        detection_confidence=detection_confidence,
        trajectory_confidence=trajectory_confidence,
        overall_confidence=overall_confidence,
    )


def _detection_confidence(validated_points: list[SparseBallDetection]) -> float:
    if not validated_points:
        return 0.0

    measured = [point for point in validated_points if not point.interpolated]
    coverage = len(measured) / max(len(validated_points), 1)
    avg_confidence = sum(point.confidence for point in measured) / max(len(measured), 1)
    count_factor = min(len(validated_points) / 8.0, 1.0)
    return max(0.0, min(1.0, avg_confidence * coverage * 0.7 + count_factor * 0.3))


def _trajectory_confidence(
    validated_points: list[SparseBallDetection],
    parabola_fit: ParabolaFit | None,
    hoop_lock: HoopLock | None,
) -> float:
    if len(validated_points) < 3 or parabola_fit is None:
        return 0.0

    fit_quality = parabola_fit.r_squared
    smoothness = _motion_smoothness(validated_points)
    continuity = min(len(validated_points) / 10.0, 1.0)
    hoop_consistency = _hoop_consistency(parabola_fit, hoop_lock)

    score = (
        0.40 * fit_quality
        + 0.30 * smoothness
        + 0.15 * continuity
        + 0.15 * hoop_consistency
    )
    return max(0.0, min(1.0, score))


def _motion_smoothness(points: list[SparseBallDetection]) -> float:
    if len(points) < 3:
        return 0.5

    velocity_changes: list[float] = []
    for index in range(2, len(points)):
        p0, p1, p2 = points[index - 2], points[index - 1], points[index]
        dt1 = max(p1.frame_index - p0.frame_index, 1)
        dt2 = max(p2.frame_index - p1.frame_index, 1)
        vx1 = (p1.x - p0.x) / dt1
        vy1 = (p1.y - p0.y) / dt1
        vx2 = (p2.x - p1.x) / dt2
        vy2 = (p2.y - p1.y) / dt2
        velocity_changes.append(abs(vx2 - vx1) + abs(vy2 - vy1))

    if not velocity_changes:
        return 0.5

    average_change = sum(velocity_changes) / len(velocity_changes)
    return max(0.0, min(1.0, 1.0 - average_change / 25.0))


def _hoop_consistency(parabola_fit: ParabolaFit, hoop_lock: HoopLock | None) -> float:
    if hoop_lock is None:
        return 0.5

    predicted_y = parabola_fit.evaluate_y(hoop_lock.center_x)
    distance = abs(predicted_y - hoop_lock.center_y)
    return max(0.0, min(1.0, 1.0 - distance / 120.0))
