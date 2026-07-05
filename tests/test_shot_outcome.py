"""Tests for make/miss classification from rim-plane crossings."""

from swishsync_cv.data import HoopLock, ParabolaFit, SparseBallDetection
from swishsync_cv.tracking.shot_outcome import classify_shot_outcome


def _point(frame: int, x: float, y: float, source: str = "measured") -> SparseBallDetection:
    return SparseBallDetection(
        frame_index=frame,
        timestamp_ms=frame * 33.0,
        x=x,
        y=y,
        confidence=0.8,
        interpolated=source != "measured",
        source=source,
    )


def _hoop() -> HoopLock:
    # squat rim+net box → ring line is the TOP: y=180; rim x-span 400..500
    return HoopLock(
        center_x=450.0,
        center_y=210.0,
        bbox_xyxy=(400.0, 180.0, 500.0, 260.0),
        confidence=0.9,
        locked_at_frame=0,
        is_locked=True,
    )


def _fit(a: float, b: float, c: float) -> ParabolaFit:
    return ParabolaFit(
        coefficients=(a, b, c),
        r_squared=0.99,
        apex_x=-b / (2 * a),
        apex_y=c - b * b / (4 * a),
        x_min=0.0,
        x_max=500.0,
        weighted_r_squared=0.99,
    )


def test_make_when_measured_descent_crosses_inside_rim_span() -> None:
    points = [_point(10, 430, 130), _point(12, 445, 170), _point(14, 455, 210)]
    outcome = classify_shot_outcome(points, _hoop())
    assert outcome.verdict == "make"
    assert outcome.method == "measured"
    assert outcome.crossing_frame == 14
    assert 400 <= outcome.crossing_x <= 500


def test_miss_when_measured_descent_crosses_outside_rim_span() -> None:
    points = [_point(10, 320, 130), _point(12, 340, 170), _point(14, 360, 210)]
    outcome = classify_shot_outcome(points, _hoop())
    assert outcome.verdict == "miss"


def test_rattle_uses_last_downward_crossing() -> None:
    # first crossing outside the span, ball pops back up, final drop inside
    points = [
        _point(10, 390, 170),
        _point(12, 395, 195),  # first downward crossing (outside)
        _point(14, 420, 165),  # back above ring line
        _point(16, 450, 200),  # final crossing inside
    ]
    outcome = classify_shot_outcome(points, _hoop())
    assert outcome.verdict == "make"
    assert outcome.crossing_frame == 16


def test_fit_fallback_make_when_tracking_stops_near_rim() -> None:
    # ball tracked to 60px above the ring, descending toward rim center;
    # parabola y = 0.01(x-350)^2 + 80 crosses y=180 at x = 350±100 → 450
    fit = _fit(0.01, -7.0, 0.01 * 350 * 350 + 80)
    points = [_point(10, 380, 89), _point(12, 410, 116), _point(14, 430, 144)]
    outcome = classify_shot_outcome(points, _hoop(), fit)
    assert outcome.verdict == "make"
    assert outcome.method == "fit"
    assert abs(outcome.crossing_x - 450.0) < 1.0


def test_fit_fallback_requires_close_approach() -> None:
    fit = _fit(0.01, -7.0, 0.01 * 350 * 350 + 80)
    points = [_point(10, 330, 20), _point(12, 340, 15)]  # 165px above ring: too far
    assert classify_shot_outcome(points, _hoop(), fit).verdict == "unknown"


def test_unknown_without_crossing_or_fit() -> None:
    points = [_point(10, 430, 130), _point(12, 445, 150)]  # never below ring
    assert classify_shot_outcome(points, _hoop()).verdict == "unknown"


def test_unknown_when_straddling_gap_too_large() -> None:
    points = [_point(10, 430, 130), _point(60, 455, 210)]  # 50-frame occlusion
    assert classify_shot_outcome(points, _hoop()).verdict == "unknown"


def test_unknown_without_hoop_lock() -> None:
    points = [_point(10, 430, 130), _point(12, 445, 210)]
    assert classify_shot_outcome(points, None).verdict == "unknown"


def test_gap_predicted_points_are_ignored() -> None:
    points = [
        _point(10, 430, 130),
        _point(12, 300, 210, source="gap_predicted"),  # would cross outside
        _point(14, 455, 210),
    ]
    assert classify_shot_outcome(points, _hoop()).verdict == "make"


def test_tall_manual_bbox_uses_bottom_as_rim_line() -> None:
    # hand-drawn box including backboard/pole: 129w x 495h → rim line = bottom
    hoop = HoopLock(
        center_x=1326.0,
        center_y=247.0,
        bbox_xyxy=(1262.0, 0.0, 1391.0, 495.0),
        confidence=1.0,
        locked_at_frame=0,
        is_locked=True,
    )
    points = [_point(10, 1300, 460), _point(12, 1320, 520)]  # crosses y=495 inside
    outcome = classify_shot_outcome(points, hoop)
    assert outcome.verdict == "make"


def test_rim_rescan_first_below_ring_point_decides() -> None:
    from swishsync_cv.tracking.shot_outcome import refine_outcome_with_rim_zone

    base = classify_shot_outcome([_point(10, 430, 130), _point(12, 445, 150)], _hoop())
    # emerged below ring inside the span → make, overriding unknown
    rim_zone = [_point(20, 450, 200), _point(22, 460, 240)]
    refined = refine_outcome_with_rim_zone(base, rim_zone, _hoop())
    assert refined.verdict == "make"
    assert refined.method == "rim_rescan"

    # emerged below ring far outside the span → miss, even if a later
    # rebound point rolls back under the net
    rim_zone = [_point(20, 320, 200), _point(24, 450, 250)]
    assert refine_outcome_with_rim_zone(base, rim_zone, _hoop()).verdict == "miss"


def test_rim_rescan_bounce_up_is_miss() -> None:
    from swishsync_cv.tracking.shot_outcome import refine_outcome_with_rim_zone

    base = classify_shot_outcome([_point(10, 430, 130), _point(12, 445, 150)], _hoop())
    # ball stays above the ring and rises: rim bounce → miss
    rim_zone = [_point(20, 460, 175), _point(22, 470, 160), _point(24, 480, 140)]
    refined = refine_outcome_with_rim_zone(base, rim_zone, _hoop())
    assert refined.verdict == "miss"


def test_rim_rescan_keeps_base_without_evidence() -> None:
    from swishsync_cv.tracking.shot_outcome import refine_outcome_with_rim_zone

    base = classify_shot_outcome(
        [_point(10, 430, 130), _point(12, 445, 170), _point(14, 455, 210)], _hoop()
    )
    assert base.verdict == "make"
    assert refine_outcome_with_rim_zone(base, [], _hoop()) == base


def test_rim_rescan_rattle_exception_inside_deeper_shortly_after() -> None:
    from swishsync_cv.tracking.shot_outcome import refine_outcome_with_rim_zone

    base = classify_shot_outcome([_point(10, 430, 130), _point(12, 445, 150)], _hoop())
    # first below-ring point on the rim edge (outside), then deeper inside → make
    rim_zone = [_point(20, 510, 185), _point(26, 455, 230)]
    assert refine_outcome_with_rim_zone(base, rim_zone, _hoop()).verdict == "make"
    # same but the inside point comes too late (rebound) → stays miss
    rim_zone = [_point(20, 510, 185), _point(40, 455, 230)]
    assert refine_outcome_with_rim_zone(base, rim_zone, _hoop()).verdict == "miss"
