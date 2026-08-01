"""Lightweight camera-space to canonical shot-space coordinate transforms."""

from __future__ import annotations

import math

from swishsync_cv.data import HoopLock, ParabolaFit, ShotAxis, ShotCandidate, SparseBallDetection


def estimate_shooter_position(validated_points: list[SparseBallDetection]) -> tuple[float, float]:
    """Use the earliest validated point as a lightweight release estimate."""

    if not validated_points:
        raise ValueError("validated_points must not be empty")
    release = validated_points[0]
    return (release.x, release.y)


def compute_shot_axis(
    shooter: tuple[float, float],
    hoop: tuple[float, float],
) -> ShotAxis:
    """Build the shooter-to-hoop axis used as the canonical horizontal reference."""

    shooter_x, shooter_y = shooter
    hoop_x, hoop_y = hoop
    axis_dx = hoop_x - shooter_x
    axis_dy = hoop_y - shooter_y
    scale = math.hypot(axis_dx, axis_dy)
    if scale < 1e-6:
        scale = 1.0
        axis_dx = 1.0
        axis_dy = 0.0
    return ShotAxis(
        shooter_x=shooter_x,
        shooter_y=shooter_y,
        hoop_x=hoop_x,
        hoop_y=hoop_y,
        axis_dx=axis_dx,
        axis_dy=axis_dy,
        scale=scale,
    )


def compute_shot_axis_from_points(
    points: list[SparseBallDetection],
    hoop_lock: HoopLock | None,
) -> ShotAxis | None:
    """Estimate shot axis from buffered or validated points."""

    if not points:
        return None

    shooter = estimate_shooter_position(points)
    if hoop_lock is not None:
        hoop = (hoop_lock.center_x, hoop_lock.center_y)
    else:
        hoop = _estimate_hoop_fallback(points, shooter)
    return compute_shot_axis(shooter, hoop)


def compute_shot_axis_from_candidate(
    candidate: ShotCandidate,
    hoop_lock: HoopLock | None,
) -> ShotAxis | None:
    """Estimate shot axis from finalized validated points."""

    if len(candidate.validated_points) < 1:
        return None

    return compute_shot_axis_from_points(candidate.validated_points, hoop_lock)


def build_normalized_collection_preview(
    candidate_points: list[SparseBallDetection],
    hoop_lock: HoopLock | None,
) -> list[tuple[float, float]]:
    """Project buffered collection points into analytical shot-space."""

    shot_axis = compute_shot_axis_from_points(candidate_points, hoop_lock)
    if shot_axis is None:
        return []
    camera_points = [(point.x, point.y) for point in candidate_points]
    return transform_arc_to_shot_space(camera_points, shot_axis)


def camera_point_to_shot_space(
    point: tuple[float, float],
    shot_axis: ShotAxis,
) -> tuple[float, float]:
    """Project one camera-space point into normalized analytical shot-space."""

    px = point[0] - shot_axis.hoop_x
    py = point[1] - shot_axis.hoop_y
    along_x, along_y = shot_axis.unit_along
    perp_x, perp_y = shot_axis.unit_perp
    x_shot = (px * along_x + py * along_y) / shot_axis.scale
    y_shot = (px * perp_x + py * perp_y) / shot_axis.scale
    return (x_shot, y_shot)


def transform_arc_to_shot_space(
    arc_points: list[tuple[float, float]],
    shot_axis: ShotAxis,
) -> list[tuple[float, float]]:
    return [camera_point_to_shot_space(point, shot_axis) for point in arc_points]


def build_camera_arc(candidate: ShotCandidate, num_points: int = 80) -> list[tuple[float, float]]:
    if candidate.parabola_fit is None:
        return [
            (point.x, point.y)
            for point in candidate.validated_points
        ]
    return candidate.parabola_fit.sample_arc(num_points=num_points)


def build_normalized_shot_view(
    candidate: ShotCandidate,
    hoop_lock: HoopLock | None,
    num_points: int = 80,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]], ShotAxis, tuple[float, float]] | None:
    """Return camera arc, normalized arc, axis, and normalized apex."""

    shot_axis = compute_shot_axis_from_candidate(candidate, hoop_lock)
    if shot_axis is None or candidate.parabola_fit is None:
        return None

    camera_arc = build_camera_arc(candidate, num_points=num_points)
    normalized_arc = transform_arc_to_shot_space(camera_arc, shot_axis)
    apex = camera_point_to_shot_space(
        (candidate.parabola_fit.apex_x, candidate.parabola_fit.apex_y),
        shot_axis,
    )
    return camera_arc, normalized_arc, shot_axis, apex


def _estimate_hoop_fallback(
    validated_points: list[SparseBallDetection],
    shooter: tuple[float, float],
) -> tuple[float, float]:
    """Infer a hoop proxy from the latest validated motion when rim lock is absent."""

    latest = validated_points[-1]
    direction_x = latest.x - shooter[0]
    direction_y = latest.y - shooter[1]
    length = math.hypot(direction_x, direction_y)
    if length < 1e-6:
        return (latest.x + 40.0, latest.y)
    return (
        shooter[0] + direction_x / length * max(length, 40.0),
        shooter[1] + direction_y / length * max(length, 40.0),
    )
