"""Tests for court-plane homography projection."""

import math

import pytest

from swishsync_cv.court.homography import CourtCalibration


def _square_calib() -> CourtCalibration:
    # image square 0..100 maps to court square 0..20 ft (simple 5x scale)
    return CourtCalibration(
        image_points=((0, 0), (100, 0), (100, 100), (0, 100)),
        court_points=((0, 0), (20, 0), (20, 20), (0, 20)),
    )


def test_corners_project_exactly():
    calib = _square_calib()
    assert calib.project(0, 0) == pytest.approx((0, 0), abs=1e-6)
    assert calib.project(100, 100) == pytest.approx((20, 20), abs=1e-6)


def test_center_projects_to_scaled_center():
    calib = _square_calib()
    cx, cy = calib.project(50, 50)
    assert cx == pytest.approx(10, abs=1e-6)
    assert cy == pytest.approx(10, abs=1e-6)


def test_requires_four_points():
    with pytest.raises(ValueError):
        CourtCalibration(image_points=((0, 0), (1, 1), (2, 2)),
                         court_points=((0, 0), (1, 1), (2, 2)))


def test_length_mismatch_rejected():
    with pytest.raises(ValueError):
        CourtCalibration(image_points=((0, 0), (1, 0), (1, 1), (0, 1)),
                         court_points=((0, 0), (1, 0), (1, 1)))


def test_json_roundtrip(tmp_path):
    calib = _square_calib()
    p = tmp_path / "calib.json"
    calib.to_json(p)
    loaded = CourtCalibration.from_json(p)
    assert loaded.project(50, 50) == pytest.approx((10, 10), abs=1e-6)
