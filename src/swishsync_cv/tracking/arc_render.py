"""Visual arc extension metadata (render-only, does not affect fitting)."""

from __future__ import annotations

from swishsync_cv.config import ShotCandidateConfig
from swishsync_cv.data import ArcRenderMetadata, HoopLock, ParabolaFit, SparseBallDetection


def compute_arc_render_metadata(
    fit: ParabolaFit,
    fit_points: list[SparseBallDetection],
    hoop_lock: HoopLock | None,
    rim_anchor_used: bool,
    config: ShotCandidateConfig,
) -> ArcRenderMetadata:
    """Derive render range and extension endpoints without changing the fit."""

    fit_x_min = float(fit.x_min)
    fit_x_max = float(fit.x_max)
    render_x_min = fit_x_min
    render_x_max = fit_x_max
    rim_center: tuple[float, float] | None = None
    visual_extension_used = False
    observed_segment_end: tuple[float, float] | None = None
    extended_segment_end: tuple[float, float] | None = None

    if fit_points:
        hoop_side_point = min(fit_points, key=lambda point: abs(point.x - _rim_x(hoop_lock)))
        observed_segment_end = (hoop_side_point.x, hoop_side_point.y)

    if hoop_lock is not None and hoop_lock.is_locked:
        rim_center = (hoop_lock.rim_center_x, hoop_lock.rim_center_y)
        rim_x = hoop_lock.rim_center_x
        fit_span = max(fit_x_max - fit_x_min, 1.0)
        rim_distance = min(abs(rim_x - fit_x_min), abs(rim_x - fit_x_max))
        max_extension = fit_span * config.max_visual_extension_ratio

        if rim_distance <= max_extension and _should_extend_to_rim(
            fit_points, hoop_lock, fit_span
        ):
            if rim_x < fit_x_min:
                render_x_min = rim_x
                visual_extension_used = True
            elif rim_x > fit_x_max:
                render_x_max = rim_x
                visual_extension_used = True

        if visual_extension_used:
            if rim_x <= fit_x_min:
                extension_start_x = fit_x_min
                extension_end_x = rim_x
            else:
                extension_start_x = fit_x_max
                extension_end_x = rim_x
            observed_segment_end = (
                extension_start_x,
                float(fit.evaluate_y(extension_start_x)),
            )
            extended_segment_end = (
                extension_end_x,
                float(fit.evaluate_y(extension_end_x)),
            )

    return ArcRenderMetadata(
        fit_x_range=(fit_x_min, fit_x_max),
        render_x_range=(render_x_min, render_x_max),
        rim_center=rim_center,
        rim_anchor_used=rim_anchor_used,
        visual_extension_used=visual_extension_used,
        observed_segment_end=observed_segment_end,
        extended_segment_end=extended_segment_end,
    )


def _rim_x(hoop_lock: HoopLock | None) -> float:
    if hoop_lock is None or not hoop_lock.is_locked:
        return 0.0
    return hoop_lock.rim_center_x


def _should_extend_to_rim(
    fit_points: list[SparseBallDetection],
    hoop_lock: HoopLock,
    fit_span: float,
) -> bool:
    """Skip extension when the last observation is still ascending far from the rim."""

    if not fit_points:
        return False
    last = max(fit_points, key=lambda point: point.frame_index)
    rim_x = hoop_lock.rim_center_x
    rim_y = hoop_lock.rim_center_y
    still_short_of_rim = (
        last.y < rim_y - 100.0 and abs(last.x - rim_x) > fit_span
    )
    return not still_short_of_rim
