"""Phase 1: story flight window sourced from trusted_flight_debug (render-only)."""

import json

from swishsync_cv.config import ShotCandidateConfig, TrustedFlightConfig
from swishsync_cv.data import ShotCandidate, SparseBallDetection, TrustedFlightSelection
from swishsync_cv.tracking import shot_finalization
from swishsync_cv.tracking.shot_finalization import finalize_shot
from swishsync_cv.tracking.shot_story import compute_shot_story
from swishsync_cv.utils.serialization import (
    fit_diagnostics_to_dict,
    parabola_fit_to_dict,
)


def _arc_points() -> list[SparseBallDetection]:
    # Clean descending-then-flat parabola arc, frames 0..8.
    return [
        SparseBallDetection(i, float(i * 33), 400.0 + i * 30.0, 300.0 - 40.0 * i + 5.0 * i * i, 0.8)
        for i in range(9)
    ]


def _long_gap_candidate() -> ShotCandidate:
    """Arc on frames 0..8, then reacquired points after a 12-frame gap.

    Gap of 12 is > trusted max_flight_gap_frames (10) but <= the fit cluster's
    reacquisition_gap_frames (15): the fit includes frames 20-21, trusted does not.
    This is exactly the case where the story window and the fit window diverge.
    """
    candidate = ShotCandidate(start_frame=0, end_frame=21, state="collecting_shot")
    candidate.candidate_points = _arc_points() + [
        SparseBallDetection(20, 660.0, 1050.0, 260.0, 0.7),
        SparseBallDetection(21, 693.0, 1080.0, 290.0, 0.7),
    ]
    candidate.continuity_points = list(candidate.candidate_points)
    return candidate


def test_story_flight_window_matches_trusted_window():
    candidate = ShotCandidate(start_frame=0, end_frame=8, state="collecting_shot")
    candidate.candidate_points = _arc_points()
    candidate.continuity_points = list(candidate.candidate_points)
    finalized = finalize_shot(candidate, ShotCandidateConfig())

    trusted = finalized.trusted_flight_debug
    assert trusted is not None and trusted.trusted
    assert finalized.story.flight_start_frame == trusted.flight_start_frame
    assert finalized.story.flight_end_frame == trusted.flight_end_frame


def test_long_gap_story_window_stops_at_gap_break():
    finalized = finalize_shot(_long_gap_candidate(), ShotCandidateConfig())

    trusted = finalized.trusted_flight_debug
    assert trusted is not None
    assert trusted.flight_end_frame == 8
    assert {e.point.frame_index for e in trusted.excluded if e.reason == "post_cluster"} == {20, 21}

    # The fit cluster (max_gap=15) still spans the gap — proves the story no
    # longer mirrors the fit window on long-gap reacquisition.
    fitted_frames = [
        p.frame_index
        for p in finalized.fit_diagnostics.points
        if p.used_in_fit and p.frame_index >= 0
    ]
    assert max(fitted_frames) == 21

    assert finalized.story.flight_end_frame == 8
    assert finalized.story.flight_start_frame == 0


def test_fit_outputs_identical_regardless_of_story_window_source(monkeypatch):
    """Byte-identical fit/confidence whether the story uses trusted or fallback."""
    normal = finalize_shot(_long_gap_candidate(), ShotCandidateConfig())

    def _empty_selection(points, hoop_lock, config, floor_margin_px=100.0):
        return TrustedFlightSelection(trusted=(), excluded=(), max_gap_frames=config.max_flight_gap_frames)

    monkeypatch.setattr(shot_finalization, "select_trusted_flight_points", _empty_selection)
    fallback = finalize_shot(_long_gap_candidate(), ShotCandidateConfig())

    assert json.dumps(parabola_fit_to_dict(normal.parabola_fit), sort_keys=True) == json.dumps(
        parabola_fit_to_dict(fallback.parabola_fit), sort_keys=True
    )
    assert json.dumps(fit_diagnostics_to_dict(normal.fit_diagnostics), sort_keys=True) == json.dumps(
        fit_diagnostics_to_dict(fallback.fit_diagnostics), sort_keys=True
    )
    assert normal.confidence == fallback.confidence

    # And the story windows DO differ — the change is confined to render metadata.
    assert normal.story.flight_end_frame == 8
    assert fallback.story.flight_end_frame == 21


def test_story_falls_back_when_trusted_flight_debug_missing():
    candidate = ShotCandidate(start_frame=0, end_frame=8, state="collecting_shot")
    candidate.candidate_points = _arc_points()
    candidate.continuity_points = list(candidate.candidate_points)
    assert candidate.trusted_flight_debug is None

    story = compute_shot_story(
        candidate,
        ShotCandidateConfig(),
        None,
        list(candidate.candidate_points),
    )
    assert story.flight_start_frame == 0
    assert story.flight_end_frame == 8


def test_trusted_selection_window_properties():
    empty = TrustedFlightSelection(trusted=(), excluded=(), max_gap_frames=10)
    assert empty.flight_start_frame is None
    assert empty.flight_end_frame is None

    points = tuple(
        SparseBallDetection(i, float(i * 33), 100.0 + i, 200.0, 0.8) for i in (3, 4, 7)
    )
    sel = TrustedFlightSelection(trusted=points, excluded=(), max_gap_frames=10)
    assert sel.flight_start_frame == 3
    assert sel.flight_end_frame == 7
