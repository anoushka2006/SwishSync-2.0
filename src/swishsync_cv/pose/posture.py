"""Shooting-form posture metrics from pose landmarks.

Pure geometry — no mediapipe dependency, fully unit-testable. Takes 2D pixel
landmarks (from pose/pose_estimator.py or any source) and computes three
shooting-form angles at the more-visible body side:

- elbow angle: shoulder-elbow-wrist (180 = straight arm at release)
- knee bend:   hip-knee-ankle    (180 = straight leg, smaller = deeper bend)
- back bend:   torso vs vertical (0 = upright, larger = leaning forward/back)

Landmark indices follow MediaPipe Pose's 33-point model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# MediaPipe Pose landmark indices
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28

# a landmark is (x, y, visibility); visibility optional (defaults to 1.0)
Landmark = tuple[float, float] | tuple[float, float, float]


@dataclass(frozen=True)
class PostureMetrics:
    """Shooting-form angles in degrees for one frame."""

    elbow_angle_deg: float | None
    knee_bend_deg: float | None
    back_bend_deg: float | None
    side: str  # "left" | "right" — which body side was measured


def joint_angle(a: Landmark, b: Landmark, c: Landmark) -> float:
    """Interior angle at vertex b (degrees), for points a-b-c in pixel space."""

    bax, bay = a[0] - b[0], a[1] - b[1]
    bcx, bcy = c[0] - b[0], c[1] - b[1]
    dot = bax * bcx + bay * bcy
    cross = bax * bcy - bay * bcx
    return math.degrees(abs(math.atan2(cross, dot)))


def back_bend_angle(shoulder: Landmark, hip: Landmark) -> float:
    """Torso lean from vertical (degrees). 0 = upright, larger = more lean."""

    # torso points hip -> shoulder; image y grows downward so "up" is (0, -1)
    tx, ty = shoulder[0] - hip[0], shoulder[1] - hip[1]
    length = math.hypot(tx, ty)
    if length < 1e-6:
        return 0.0
    # angle between torso vector and vertical-up (0, -1)
    cos_theta = (-ty) / length
    cos_theta = max(-1.0, min(1.0, cos_theta))
    return math.degrees(math.acos(cos_theta))


def _visibility(point: Landmark) -> float:
    return point[2] if len(point) > 2 else 1.0


def _side_visibility(landmarks: dict[int, Landmark], indices: tuple[int, ...]) -> float:
    return sum(_visibility(landmarks[i]) for i in indices if i in landmarks)


def compute_posture(
    landmarks: dict[int, Landmark],
    min_visibility: float = 0.5,
) -> PostureMetrics:
    """Pick the more-visible body side and compute the three form angles."""

    left = (L_SHOULDER, L_ELBOW, L_WRIST, L_HIP, L_KNEE, L_ANKLE)
    right = (R_SHOULDER, R_ELBOW, R_WRIST, R_HIP, R_KNEE, R_ANKLE)
    use_left = _side_visibility(landmarks, left) >= _side_visibility(landmarks, right)
    side_idx = left if use_left else right
    shoulder, elbow, wrist, hip, knee, ankle = side_idx

    def angle_if_visible(getter, *idx):
        pts = [landmarks.get(i) for i in idx]
        if any(p is None or _visibility(p) < min_visibility for p in pts):
            return None
        return getter(*pts)

    return PostureMetrics(
        elbow_angle_deg=angle_if_visible(joint_angle, shoulder, elbow, wrist),
        knee_bend_deg=angle_if_visible(joint_angle, hip, knee, ankle),
        back_bend_deg=angle_if_visible(back_bend_angle, shoulder, hip),
        side="left" if use_left else "right",
    )
