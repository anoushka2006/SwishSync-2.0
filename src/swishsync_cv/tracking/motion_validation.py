"""Lightweight motion validation for sparse shot points."""

from __future__ import annotations

from swishsync_cv.config import ShotCandidateConfig
from swishsync_cv.data import ParabolaFit, SparseBallDetection


def validate_point(
    candidate_point: SparseBallDetection,
    validated_points: list[SparseBallDetection],
    config: ShotCandidateConfig,
    parabola_fit: ParabolaFit | None = None,
) -> bool:
    """Return True when a raw point passes basic physical plausibility checks."""

    if not validated_points:
        return True

    previous = validated_points[-1]
    dx = abs(candidate_point.x - previous.x)
    if dx > config.max_horizontal_jump_px:
        return False

    if len(validated_points) >= 2:
        prior = validated_points[-2]
        dt_prev = max(previous.frame_index - prior.frame_index, 1)
        dt_curr = max(candidate_point.frame_index - previous.frame_index, 1)
        vy_prev = (previous.y - prior.y) / dt_prev
        vy_curr = (candidate_point.y - previous.y) / dt_curr
        vertical_accel = abs(vy_curr - vy_prev)
        if vertical_accel > config.max_vertical_accel_px:
            return False

        if _is_before_apex(validated_points) and _is_upward_reversal(vy_prev, vy_curr):
            return False

    if parabola_fit is not None and len(validated_points) >= config.min_validated_points_for_fit:
        predicted_y = parabola_fit.evaluate_y(candidate_point.x)
        if abs(candidate_point.y - predicted_y) > config.max_parabola_deviation_px:
            return False

    return True


def validate_shot_points(
    candidate_points: list[SparseBallDetection],
    config: ShotCandidateConfig,
) -> list[SparseBallDetection]:
    """Filter noisy outliers from a completed candidate buffer."""

    validated: list[SparseBallDetection] = []
    for point in candidate_points:
        if validate_point(
            candidate_point=point,
            validated_points=validated,
            config=config,
            parabola_fit=None,
        ):
            validated.append(point)
    return validated


def _is_before_apex(points: list[SparseBallDetection]) -> bool:
    if len(points) < 2:
        return True
    ys = [point.y for point in points]
    return min(ys) == ys[-1]


def _is_upward_reversal(previous_vy: float, current_vy: float) -> bool:
    """Detect a sudden downward-to-upward flip before apex (y increases downward)."""

    return previous_vy > 1.0 and current_vy < -1.0
