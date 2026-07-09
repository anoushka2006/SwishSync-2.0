"""End-to-end smoke test for the platform scaffold — no video, no torch.

Builds a synthetic ball Track along a known parabola, runs it through the
registered ShotEventDetector, and asserts one SHOT event with a sane fit. Proves
the IR + registry + event interface + legacy-fit adapter all connect.

Run:
    PYTHONPATH=src python scripts/smoke_platform.py
"""

from __future__ import annotations

import sys

from swishsync.core.registry import EVENT_DETECTORS, get
from swishsync.core.schemas import EventType, ObjectClass, Track, TrackState


def _synthetic_ball_track() -> Track:
    # y = a*x^2 + b*x + c ; image y grows downward so apex is a minimum-y here.
    a, b, c = 0.02, -6.0, 500.0
    states: list[TrackState] = []
    for i, x in enumerate(range(100, 400, 12)):
        y = a * x * x + b * x + c
        # 6px box centered on (x, y)
        states.append(
            TrackState(
                frame=i,
                t_ms=i * 33.0,
                bbox_xyxy=(x - 3, y - 3, x + 3, y + 3),
                conf=0.85,
            )
        )
    return Track(track_id=1, cls=ObjectClass.BALL, states=states)


def main() -> int:
    detector_cls = get(EVENT_DETECTORS, "shot")
    detector = detector_cls()

    ball = _synthetic_ball_track()
    events = detector.detect([ball])

    assert len(events) == 1, f"expected 1 shot event, got {len(events)}"
    ev = events[0]
    assert ev.type == EventType.SHOT
    assert ev.actors == [1]
    fit = ev.payload["parabola_fit"]
    a, b, c = fit.coefficients
    assert abs(a - 0.02) < 0.01, f"a coefficient off: {a}"
    assert fit.weighted_r_squared > 0.95, f"bad fit R^2: {fit.weighted_r_squared}"

    print("SMOKE OK")
    print(f"  registered event detectors: {sorted(EVENT_DETECTORS)}")
    print(f"  shot event: actors={ev.actors} conf={ev.confidence:.4f}")
    print(f"  fit coeffs (a,b,c)=({a:.4f},{b:.3f},{c:.1f}) "
          f"rmse={fit.weighted_residual_rmse:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
