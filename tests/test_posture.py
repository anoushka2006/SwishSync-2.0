"""Tests for pure posture-angle math (no mediapipe)."""

import math

from swishsync_cv.pose.posture import (
    back_bend_angle,
    compute_posture,
    joint_angle,
)


def test_joint_angle_straight_is_180():
    assert abs(joint_angle((0, 0), (1, 0), (2, 0)) - 180.0) < 1e-6


def test_joint_angle_right_angle_is_90():
    assert abs(joint_angle((0, 1), (0, 0), (1, 0)) - 90.0) < 1e-6


def test_joint_angle_is_orientation_independent():
    # 90 deg regardless of which way the corner points
    assert abs(joint_angle((1, 0), (0, 0), (0, -1)) - 90.0) < 1e-6


def test_back_bend_upright_is_zero():
    # shoulder directly above hip (image y grows down) → upright
    assert back_bend_angle((100, 50), (100, 200)) < 1e-6


def test_back_bend_45_degrees():
    # shoulder up-and-right of hip by equal amounts → 45 deg lean
    assert abs(back_bend_angle((150, 150), (100, 200)) - 45.0) < 1e-6


def test_compute_posture_picks_more_visible_side():
    # right side fully visible, left side low visibility → measures right
    lm = {
        12: (200, 100, 0.9), 14: (220, 150, 0.9), 16: (200, 200, 0.9),  # R arm bent
        24: (200, 260, 0.9), 26: (200, 360, 0.9), 28: (200, 460, 0.9),  # R leg straight
        11: (100, 100, 0.1), 13: (100, 150, 0.1), 15: (100, 200, 0.1),
        23: (100, 260, 0.1), 25: (100, 360, 0.1), 27: (100, 460, 0.1),
    }
    m = compute_posture(lm)
    assert m.side == "right"
    assert m.knee_bend_deg is not None and abs(m.knee_bend_deg - 180.0) < 1e-6
    assert m.elbow_angle_deg is not None  # bent arm, some angle < 180


def test_compute_posture_none_when_side_not_visible():
    lm = {11: (100, 100, 0.1), 13: (100, 150, 0.1), 15: (100, 200, 0.1)}
    m = compute_posture(lm)
    assert m.elbow_angle_deg is None
    assert m.knee_bend_deg is None
