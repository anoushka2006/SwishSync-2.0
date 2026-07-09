"""Pytest smoke tests for the platform scaffold."""

from __future__ import annotations

from swishsync.core.registry import EVENT_DETECTORS, get
from swishsync.core.schemas import EventType, ObjectClass, Track, TrackState


def _synthetic_ball_track() -> Track:
    a, b, c = 0.02, -6.0, 500.0
    states = [
        TrackState(
            frame=i,
            t_ms=i * 33.0,
            bbox_xyxy=(x - 3, a * x * x + b * x + c - 3, x + 3, a * x * x + b * x + c + 3),
            conf=0.85,
        )
        for i, x in enumerate(range(100, 400, 12))
    ]
    return Track(track_id=1, cls=ObjectClass.BALL, states=states)


def test_shot_detector_is_registered():
    assert "shot" in EVENT_DETECTORS


def test_shot_event_from_synthetic_track():
    detector = get(EVENT_DETECTORS, "shot")()
    events = detector.detect([_synthetic_ball_track()])
    assert len(events) == 1
    ev = events[0]
    assert ev.type == EventType.SHOT
    assert ev.actors == [1]
    fit = ev.payload["parabola_fit"]
    assert abs(fit.coefficients[0] - 0.02) < 0.01
    assert fit.weighted_r_squared > 0.95


def test_short_track_yields_no_event():
    short = Track(
        track_id=2,
        cls=ObjectClass.BALL,
        states=[TrackState(frame=0, t_ms=0.0, bbox_xyxy=(0, 0, 6, 6), conf=0.8)],
    )
    detector = get(EVENT_DETECTORS, "shot")()
    assert detector.detect([short]) == []


def test_non_ball_track_ignored():
    player = Track(
        track_id=3,
        cls=ObjectClass.PLAYER,
        states=[
            TrackState(frame=i, t_ms=i * 33.0, bbox_xyxy=(i, i, i + 6, i + 6), conf=0.9)
            for i in range(10)
        ],
    )
    detector = get(EVENT_DETECTORS, "shot")()
    assert detector.detect([player]) == []
