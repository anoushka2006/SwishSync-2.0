"""Simple trajectory panel: collection dots and one finalized arc."""

from __future__ import annotations

import cv2
import numpy as np

from swishsync_cv.data import FitDiagnostics, HoopLock, PointDiagnostic, ShotCandidate
from swishsync_cv.tracking.gap_recovery import continuity_track
from swishsync_cv.tracking.parabola import confidence_tier
from swishsync_cv.visualization.arc_drawing import draw_finalized_arc
from swishsync_cv.visualization.continuity_drawing import GAP_PREDICTED_COLOR, draw_continuity_track

PANEL_TITLE = "Shot Trajectory"
TEXT_COLOR = (235, 235, 235)
COLLECTING_PATH_COLOR = (130, 130, 170)
FINAL_ARC_COLOR = (80, 220, 255)
APEX_COLOR = (255, 180, 80)
HOOP_COLOR = (80, 120, 255)
HIGH_CONF_COLOR = (80, 220, 120)
MEDIUM_CONF_COLOR = (80, 200, 255)
LOW_CONF_COLOR = (100, 100, 255)
OUTLIER_COLOR = (80, 80, 255)


def render_trajectory_panel(
    frame_size: tuple[int, int],
    collecting_shot: ShotCandidate | None,
    display_shot: ShotCandidate | None,
    hoop_lock: HoopLock | None,
    lifecycle_state: str,
    candidate_point_count: int,
    background_color: tuple[int, int, int] = (24, 24, 28),
) -> np.ndarray:
    """Render camera-space collection preview or one finalized parabola."""

    width, height = frame_size
    panel = np.full((height, width, 3), background_color, dtype=np.uint8)

    _draw_header(panel, lifecycle_state, candidate_point_count)

    if hoop_lock is not None and hoop_lock.is_locked:
        center = (int(round(hoop_lock.center_x)), int(round(hoop_lock.center_y)))
        cv2.circle(panel, center, 10, HOOP_COLOR, 2)
        rim = (int(round(hoop_lock.rim_center_x)), int(round(hoop_lock.rim_center_y)))
        cv2.drawMarker(
            panel,
            rim,
            HOOP_COLOR,
            markerType=cv2.MARKER_TILTED_CROSS,
            markerSize=12,
            thickness=2,
        )

    if collecting_shot is not None and collecting_shot.state == "collecting_shot":
        _draw_collection_preview(panel, collecting_shot)

    if display_shot is not None and display_shot.insufficient_points_for_fit:
        _draw_insufficient_shot(panel, display_shot)
    elif display_shot is not None and display_shot.parabola_fit is not None:
        _draw_finalized_shot(panel, display_shot)

    if (
        collecting_shot is None
        and (
            display_shot is None
            or (
                display_shot.parabola_fit is None
                and not display_shot.insufficient_points_for_fit
            )
        )
    ):
        cv2.putText(
            panel,
            "Awaiting shot...",
            (16, height // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            TEXT_COLOR,
            1,
            cv2.LINE_AA,
        )

    return panel


def compose_dual_pane(
    left_panel: np.ndarray,
    right_panel: np.ndarray,
) -> np.ndarray:
    if left_panel.shape != right_panel.shape:
        right_panel = cv2.resize(right_panel, (left_panel.shape[1], left_panel.shape[0]))
    return np.hstack([left_panel, right_panel])


def _draw_header(panel: np.ndarray, lifecycle_state: str, candidate_point_count: int) -> None:
    cv2.putText(
        panel,
        PANEL_TITLE,
        (16, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        TEXT_COLOR,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        f"state={lifecycle_state.upper()}  points={candidate_point_count}",
        (16, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        TEXT_COLOR,
        1,
        cv2.LINE_AA,
    )


def _draw_collection_preview(panel: np.ndarray, collecting_shot: ShotCandidate) -> None:
    draw_continuity_track(panel, collecting_shot)

    cv2.putText(
        panel,
        "COLLECTING (no fit)",
        (16, 76),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        TEXT_COLOR,
        1,
        cv2.LINE_AA,
    )


def _draw_insufficient_shot(panel: np.ndarray, display_shot: ShotCandidate) -> None:
    draw_continuity_track(panel, display_shot)

    cv2.putText(
        panel,
        "INSUFFICIENT POINTS FOR FIT",
        (16, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        OUTLIER_COLOR,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        f"measured={len(display_shot.candidate_points)} need=5",
        (16, 122),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        TEXT_COLOR,
        1,
        cv2.LINE_AA,
    )


def _draw_finalized_shot(panel: np.ndarray, display_shot: ShotCandidate) -> None:
    fit = display_shot.parabola_fit
    if fit is None:
        return

    draw_finalized_arc(panel, display_shot)
    draw_continuity_track(panel, display_shot)

    diagnostics = display_shot.fit_diagnostics
    if diagnostics is not None:
        for point in diagnostics.points:
            if point.frame_index < 0:
                cv2.drawMarker(
                    panel,
                    (int(round(point.x)), int(round(point.y))),
                    HOOP_COLOR,
                    markerType=cv2.MARKER_TILTED_CROSS,
                    markerSize=12,
                    thickness=2,
                )
                continue
            _draw_confidence_point(
                panel,
                int(round(point.x)),
                int(round(point.y)),
                point.confidence,
                is_outlier=point.is_outlier or point.excluded_from_fit,
            )
    else:
        for point in display_shot.candidate_points:
            _draw_confidence_point(
                panel,
                int(round(point.x)),
                int(round(point.y)),
                point.confidence,
                is_outlier=False,
            )

    for point in display_shot.excluded_debug_points:
        _draw_confidence_point(
            panel,
            int(round(point.x)),
            int(round(point.y)),
            point.confidence,
            is_outlier=True,
        )

    apex = (int(round(fit.apex_x)), int(round(fit.apex_y)))
    cv2.circle(panel, apex, 6, APEX_COLOR, -1)
    _draw_fit_diagnostics_hud(panel, display_shot, diagnostics)


def _draw_confidence_point(
    panel: np.ndarray,
    x: int,
    y: int,
    confidence: float,
    is_outlier: bool,
) -> None:
    color = _confidence_color(confidence)
    cv2.circle(panel, (x, y), 4, color, -1)
    if is_outlier:
        cv2.drawMarker(
            panel,
            (x, y),
            OUTLIER_COLOR,
            markerType=cv2.MARKER_TILTED_CROSS,
            markerSize=10,
            thickness=2,
        )


def _confidence_color(confidence: float) -> tuple[int, int, int]:
    tier = confidence_tier(confidence)
    if tier == "high":
        return HIGH_CONF_COLOR
    if tier == "medium":
        return MEDIUM_CONF_COLOR
    return LOW_CONF_COLOR


def _draw_fit_diagnostics_hud(
    panel: np.ndarray,
    display_shot: ShotCandidate,
    diagnostics: FitDiagnostics | None,
) -> None:
    fit = display_shot.parabola_fit
    if fit is None:
        return

    lines = [f"FINALIZED weighted r2={fit.weighted_r_squared:.2f}"]
    if diagnostics is not None:
        lines.extend(
            [
                f"points={diagnostics.point_count}",
                f"avg detection conf={diagnostics.average_detection_confidence:.2f}",
                f"weighted rmse={diagnostics.weighted_residual_rmse:.1f}px",
                f"outliers={diagnostics.outlier_count}",
            ]
        )
    if display_shot.confidence is not None:
        lines.append(
            f"trajectory confidence={int(round(display_shot.confidence.trajectory_confidence * 100))}%"
        )
        if display_shot.confidence.gap_predicted_count:
            lines.append(
                f"gap predicted={display_shot.confidence.gap_predicted_count}"
                f" coverage={display_shot.confidence.continuity_coverage:.0%}"
            )

    y_offset = 76
    for line in lines:
        cv2.putText(
            panel,
            line,
            (16, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            TEXT_COLOR,
            1,
            cv2.LINE_AA,
        )
        y_offset += 18

    legend_x = panel.shape[1] - 170
    cv2.putText(panel, "high", (legend_x, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.4, HIGH_CONF_COLOR, 1)
    cv2.putText(panel, "med", (legend_x, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.4, MEDIUM_CONF_COLOR, 1)
    cv2.putText(panel, "low", (legend_x, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.4, LOW_CONF_COLOR, 1)
    cv2.putText(panel, "outlier", (legend_x, 124), cv2.FONT_HERSHEY_SIMPLEX, 0.4, OUTLIER_COLOR, 1)
    cv2.putText(panel, "gap", (legend_x, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.4, GAP_PREDICTED_COLOR, 1)


def _draw_dotted_line(
    panel: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    x1, y1 = start
    x2, y2 = end
    length = int(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)
    if length <= 0:
        return
    steps = max(length // 6, 1)
    for step in range(0, steps, 2):
        t0 = step / steps
        t1 = min((step + 1) / steps, 1.0)
        p0 = (int(x1 + (x2 - x1) * t0), int(y1 + (y2 - y1) * t0))
        p1 = (int(x1 + (x2 - x1) * t1), int(y1 + (y2 - y1) * t1))
        cv2.line(panel, p0, p1, color, 1, cv2.LINE_AA)
