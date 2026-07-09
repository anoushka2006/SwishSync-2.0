"""ShotEventDetector — the first EventDetector.

This is the Phase-A migration adapter. It does NOT reimplement the shot fit. It
takes the platform's ball Track, adapts it to the legacy
`swishsync_cv` point type, calls the existing, proven robust parabola fit, and
wraps the result back into a platform Event. That reuse is deliberate: the exit
gate for Phase A is byte-identical shot output through the new pipeline, which is
only meaningful if the same fit code runs.

When the legacy package is ported into `swishsync.events.fit`, only the adapter
below changes; the interface and IR stay put.
"""

from __future__ import annotations

from swishsync.core.registry import register_event_detector
from swishsync.core.schemas import (
    Event,
    EventType,
    ObjectClass,
    ParabolaFit,
    Track,
    WorldState,
)
from swishsync.events.interfaces import EventDetector

# Legacy fit is imported lazily/guarded so this module still imports on a machine
# where swishsync_cv isn't installed (e.g. a docs build). The error only fires if
# you actually try to detect a shot without the engine present.
try:
    from swishsync_cv.data import SparseBallDetection as _LegacyPoint
    from swishsync_cv.tracking.parabola import (
        fit_weighted_parabola_robust as _legacy_fit,
    )

    _LEGACY_AVAILABLE = True
except Exception as _exc:  # pragma: no cover
    _LEGACY_AVAILABLE = False
    _LEGACY_IMPORT_ERROR = _exc


def _track_to_legacy_points(ball: Track) -> list["_LegacyPoint"]:
    points: list[_LegacyPoint] = []
    for state in ball.states:
        cx, cy = state.center
        points.append(
            _LegacyPoint(
                frame_index=state.frame,
                timestamp_ms=state.t_ms,
                x=cx,
                y=cy,
                confidence=state.conf,
            )
        )
    return points


def _legacy_fit_to_ir(fit) -> ParabolaFit:
    return ParabolaFit(
        coefficients=tuple(fit.coefficients),
        r_squared=fit.r_squared,
        apex_x=fit.apex_x,
        apex_y=fit.apex_y,
        x_min=fit.x_min,
        x_max=fit.x_max,
        weighted_r_squared=fit.weighted_r_squared,
        weighted_residual_rmse=fit.weighted_residual_rmse,
    )


@register_event_detector("shot")
class ShotEventDetector(EventDetector):
    """Fit one parabola to the ball track and emit a SHOT event.

    min_flight_points mirrors the legacy fitter's >=3 floor and the arc quality
    bar in CLAUDE.md (a shot with >=4 flight points should fit).
    """

    def __init__(self, min_flight_points: int = 3) -> None:
        self.min_flight_points = min_flight_points

    def detect(
        self,
        tracks: list[Track],
        world: list[WorldState] | None = None,
    ) -> list[Event]:
        if not _LEGACY_AVAILABLE:  # pragma: no cover
            raise RuntimeError(
                "swishsync_cv fit engine not importable; cannot detect shots. "
                f"Original import error: {_LEGACY_IMPORT_ERROR!r}"
            )

        ball_tracks = [t for t in tracks if t.cls == ObjectClass.BALL]
        events: list[Event] = []

        for ball in ball_tracks:
            if len(ball.states) < self.min_flight_points:
                continue

            points = _track_to_legacy_points(ball)
            result = _legacy_fit(points)
            if result is None:
                continue

            fit, _diagnostics = result
            frames = [s for s in ball.states]
            events.append(
                Event(
                    type=EventType.SHOT,
                    t_start_ms=frames[0].t_ms,
                    t_end_ms=frames[-1].t_ms,
                    actors=[ball.track_id],
                    confidence=fit.weighted_r_squared,
                    evidence={
                        "flight_points": len(points),
                        "weighted_residual_rmse": fit.weighted_residual_rmse,
                        "frame_range": [frames[0].frame, frames[-1].frame],
                    },
                    payload={"parabola_fit": _legacy_fit_to_ir(fit)},
                )
            )

        return events
