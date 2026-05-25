import numpy as np

from swishsync_cv.config import ShotCandidateConfig
from swishsync_cv.data import ShotCandidate, SparseBallDetection
from swishsync_cv.tracking.parabola import fit_weighted_parabola
from swishsync_cv.tracking.shot_candidate import ShotCandidateManager
from swishsync_cv.tracking.shot_finalization import finalize_shot
from swishsync_cv.visualization.trajectory_panel import compose_dual_pane, render_trajectory_panel


def test_finalize_shot_fits_once_from_buffered_points():
    candidate = ShotCandidate(start_frame=0, state="collecting_shot")
    candidate.candidate_points = [
        SparseBallDetection(0, 0.0, 10.0, 80.0, 0.9),
        SparseBallDetection(1, 33.3, 30.0, 60.0, 0.9),
        SparseBallDetection(2, 66.6, 50.0, 45.0, 0.9),
        SparseBallDetection(3, 99.9, 70.0, 40.0, 0.9),
    ]
    candidate.end_frame = 3

    finalized = finalize_shot(candidate, ShotCandidateConfig())

    assert finalized.state == "shot_finalized"
    assert finalized.parabola_fit is not None
    assert finalized.fit_diagnostics is not None
    assert finalized.confidence is not None
    assert finalized.fit_diagnostics.point_count == 4


def test_manager_does_not_fit_during_collection():
    manager = ShotCandidateManager(ShotCandidateConfig(), frame_height=100)
    points = [
        SparseBallDetection(0, 0.0, 20.0, 80.0, 0.9),
        SparseBallDetection(1, 33.3, 22.0, 70.0, 0.9),
        SparseBallDetection(2, 66.6, 24.0, 52.0, 0.9),
        SparseBallDetection(3, 99.9, 26.0, 45.0, 0.9),
    ]
    for point in points:
        manager.update(point.frame_index, point)

    assert manager.lifecycle_state == "collecting_shot"
    assert manager.active is not None
    assert manager.active.parabola_fit is None


def test_manager_resets_after_finalization():
    manager = ShotCandidateManager(ShotCandidateConfig(), frame_height=100)
    manager.active = ShotCandidate(start_frame=0, state="collecting_shot")
    manager.active.candidate_points = [
        SparseBallDetection(0, 0.0, 10.0, 80.0, 0.9),
        SparseBallDetection(1, 33.3, 30.0, 60.0, 0.9),
        SparseBallDetection(2, 66.6, 50.0, 45.0, 0.9),
        SparseBallDetection(3, 99.9, 70.0, 40.0, 0.9),
    ]

    finalized = manager.finalize()

    assert manager.active is None
    assert manager.lifecycle_state == "idle"
    assert finalized is not None
    assert finalized.parabola_fit is not None
    assert manager.display_shot is finalized


def test_trajectory_panel_renders_one_finalized_arc():
    display = ShotCandidate(start_frame=0, end_frame=3, state="shot_finalized")
    display.candidate_points = [
        SparseBallDetection(0, 0.0, 10.0, 80.0, 0.9),
        SparseBallDetection(1, 33.3, 30.0, 60.0, 0.9),
        SparseBallDetection(2, 66.6, 50.0, 45.0, 0.9),
        SparseBallDetection(3, 99.9, 70.0, 40.0, 0.9),
    ]
    result = fit_weighted_parabola(display.candidate_points)
    assert result is not None
    display.parabola_fit, display.fit_diagnostics = result
    from swishsync_cv.tracking.confidence import score_shot_confidence

    display.confidence = score_shot_confidence(display)

    panel = render_trajectory_panel(
        frame_size=(160, 120),
        collecting_shot=None,
        display_shot=display,
        hoop_lock=None,
        lifecycle_state="idle",
        candidate_point_count=0,
    )
    assert panel.shape == (120, 160, 3)
    assert np.count_nonzero(panel) > 0


def test_compose_dual_pane_doubles_width():
    left = np.zeros((64, 96, 3), dtype=np.uint8)
    right = np.zeros((64, 96, 3), dtype=np.uint8)
    combined = compose_dual_pane(left, right)
    assert combined.shape == (64, 192, 3)
