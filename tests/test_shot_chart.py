"""Tests for half-court shot-chart rendering."""

import numpy as np

from swishsync_cv.court.shot_chart import ShotMark, render_shot_chart


def test_render_returns_image():
    img = render_shot_chart([ShotMark(15, 5, True), ShotMark(20, -8, False)])
    assert img.ndim == 3 and img.shape[2] == 3
    assert img.shape[0] > 0 and img.shape[1] > 0


def test_make_and_miss_pixels_present():
    # a make (green) and miss (red) mark should paint their colors
    img = render_shot_chart([ShotMark(10, 0, True)])
    green = np.all(np.abs(img.astype(int) - [90, 210, 90]) < 30, axis=2)
    assert green.any()
    img2 = render_shot_chart([ShotMark(10, 0, False)])
    red = np.all(np.abs(img2.astype(int) - [80, 80, 230]) < 30, axis=2)
    assert red.any()


def test_out_of_bounds_marks_skipped_without_error():
    img = render_shot_chart([ShotMark(999, 999, True)])
    assert img.shape[0] > 0
