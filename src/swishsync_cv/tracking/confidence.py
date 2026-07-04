"""Composite confidence scoring for reconstructed shots."""

from __future__ import annotations

from swishsync_cv.data import (
    ConfidenceScores,
    FitDiagnostics,
    ParabolaFit,
    ShotCandidate,
    SparseBallDetection,
    effective_point_source,
    is_measured_detection,
)
from swishsync_cv.tracking.gap_recovery import continuity_track
from swishsync_cv.tracking.parabola import analyze_trajectory_completeness


def score_shot_confidence(candidate: ShotCandidate) -> ConfidenceScores:
    """Compute detection, trajectory, and overall confidence for a finalized shot."""

    points = candidate.candidate_points
    track = continuity_track(candidate)
    gap_predicted_count = sum(
        1 for point in track if effective_point_source(point) == "gap_predicted"
    )
    measured_in_track = sum(1 for point in track if is_measured_detection(point))
    continuity_coverage = measured_in_track / max(len(track), 1)

    detection_confidence = _detection_confidence(points)
    trajectory_confidence = _trajectory_confidence(
        points=points,
        parabola_fit=candidate.parabola_fit,
        fit_diagnostics=candidate.fit_diagnostics,
        trajectory_incomplete=analyze_trajectory_completeness(points)[0],
    )
    overall_confidence = 0.40 * detection_confidence + 0.60 * trajectory_confidence
    return ConfidenceScores(
        detection_confidence=detection_confidence,
        trajectory_confidence=trajectory_confidence,
        overall_confidence=overall_confidence,
        continuity_coverage=continuity_coverage,
        gap_predicted_count=gap_predicted_count,
    )


def _detection_confidence(points: list[SparseBallDetection]) -> float:
    if not points:
        return 0.0

    measured = [point for point in points if not point.interpolated]
    coverage = len(measured) / max(len(points), 1)
    avg_confidence = sum(point.confidence for point in measured) / max(len(measured), 1)
    count_factor = min(len(points) / 8.0, 1.0)
    return max(0.0, min(1.0, avg_confidence * coverage * 0.65 + count_factor * 0.35))


def _trajectory_confidence(
    points: list[SparseBallDetection],
    parabola_fit: ParabolaFit | None,
    fit_diagnostics: FitDiagnostics | None,
    trajectory_incomplete: bool = False,
) -> float:
    if len(points) < 3 or parabola_fit is None or fit_diagnostics is None:
        return 0.0

    fit_quality = parabola_fit.weighted_r_squared
    residual_quality = max(0.0, min(1.0, 1.0 - fit_diagnostics.weighted_residual_rmse / 45.0))
    continuity = min(len(points) / 10.0, 1.0)
    smoothness = _motion_smoothness(points)
    outlier_penalty = max(
        0.0,
        1.0 - fit_diagnostics.outlier_count / max(len(points), 1),
    )

    score = (
        0.30 * fit_quality
        + 0.30 * residual_quality
        + 0.20 * smoothness
        + 0.10 * continuity
        + 0.10 * outlier_penalty
    )
    if trajectory_incomplete:
        score *= 0.80
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
