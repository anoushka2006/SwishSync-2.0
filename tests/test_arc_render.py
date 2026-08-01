from swishsync_cv.config import ShotCandidateConfig
from swishsync_cv.data import HoopLock, ParabolaFit, SparseBallDetection
from swishsync_cv.tracking.arc_render import compute_arc_render_metadata


def _hoop() -> HoopLock:
    return HoopLock(
        center_x=500.0,
        center_y=365.0,
        bbox_xyxy=(430.0, 320.0, 570.0, 410.0),
        confidence=1.0,
        locked_at_frame=0,
        is_locked=True,
    )


def _clip_b_fit() -> ParabolaFit:
    return ParabolaFit(
        coefficients=(0.004111, -5.0, 2000.0),
        r_squared=1.0,
        apex_x=742.0,
        apex_y=142.0,
        x_min=698.0,
        x_max=1041.0,
        weighted_r_squared=1.0,
        weighted_residual_rmse=0.6,
    )


def test_visual_extension_toward_rim_when_beyond_fit_range():
    fit_points = [
        SparseBallDetection(78, 0.0, 1041.0, 320.0, 0.64),
        SparseBallDetection(94, 0.0, 698.0, 145.0, 0.46),
    ]
    metadata = compute_arc_render_metadata(
        _clip_b_fit(),
        fit_points,
        _hoop(),
        rim_anchor_used=False,
        config=ShotCandidateConfig(),
    )

    assert metadata.visual_extension_used is True
    assert metadata.render_x_range[0] == 500.0
    assert metadata.fit_x_range == (698.0, 1041.0)
    assert metadata.rim_anchor_used is False
    assert metadata.extended_segment_end is not None
    assert metadata.extended_segment_end[0] == 500.0


def test_no_extension_when_rim_inside_fit_range():
    fit = ParabolaFit(
        coefficients=(0.01, -5.0, 1000.0),
        r_squared=1.0,
        apex_x=500.0,
        apex_y=100.0,
        x_min=450.0,
        x_max=900.0,
    )
    metadata = compute_arc_render_metadata(
        fit,
        [SparseBallDetection(0, 0.0, 450.0, 200.0, 0.9)],
        _hoop(),
        rim_anchor_used=False,
        config=ShotCandidateConfig(),
    )

    assert metadata.visual_extension_used is False
    assert metadata.render_x_range == metadata.fit_x_range


def test_no_extension_when_observed_flight_still_short_of_rim():
    """Side-angle clip D: ascending segment far from rim should not extend."""
    fit = ParabolaFit(
        coefficients=(0.000663, -1.593, 990.5),
        r_squared=1.0,
        apex_x=1201.0,
        apex_y=33.0,
        x_min=465.0,
        x_max=713.0,
    )
    hoop = HoopLock(
        center_x=1294.5,
        center_y=401.5,
        bbox_xyxy=(1231.0, 312.0, 1358.0, 491.0),
        confidence=1.0,
        locked_at_frame=0,
        is_locked=True,
    )
    fit_points = [
        SparseBallDetection(62, 0.0, 465.0, 392.0, 0.66),
        SparseBallDetection(68, 0.0, 713.0, 192.0, 0.30),
    ]
    metadata = compute_arc_render_metadata(
        fit,
        fit_points,
        hoop,
        rim_anchor_used=False,
        config=ShotCandidateConfig(),
    )

    assert metadata.visual_extension_used is False
    assert metadata.render_x_range == metadata.fit_x_range


def test_no_extension_when_rim_too_far():
    fit = ParabolaFit(
        coefficients=(0.01, -5.0, 1000.0),
        r_squared=1.0,
        apex_x=500.0,
        apex_y=100.0,
        x_min=900.0,
        x_max=950.0,
    )
    metadata = compute_arc_render_metadata(
        fit,
        [SparseBallDetection(0, 0.0, 900.0, 200.0, 0.9)],
        _hoop(),
        rim_anchor_used=False,
        config=ShotCandidateConfig(max_visual_extension_ratio=0.1),
    )

    assert metadata.visual_extension_used is False
