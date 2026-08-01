"""Court-plane homography: map image pixels to real court coordinates.

A static camera sees the court as a plane, so 4+ known image<->court point
pairs define a homography that projects any ground point (e.g. the shooter's
feet) into court coordinates (feet). Calibration is per-camera-setup and done
once via the confirm-step tool (scripts/calibrate_court.py); this module is the
pure math + storage.

Court coordinate convention (half court, feet): origin at the hoop's floor
point, +x toward half-court, +y toward the right sideline (camera's right).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class CourtCalibration:
    """Image<->court point correspondences and the derived homography."""

    image_points: tuple[tuple[float, float], ...]
    court_points: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        if len(self.image_points) != len(self.court_points):
            raise ValueError("image_points and court_points must be the same length")
        if len(self.image_points) < 4:
            raise ValueError("need >= 4 point pairs for a homography")

    def _homography(self) -> np.ndarray:
        import cv2

        src = np.array(self.image_points, dtype=np.float64)
        dst = np.array(self.court_points, dtype=np.float64)
        matrix, _ = cv2.findHomography(src, dst, method=0)
        if matrix is None:
            raise ValueError("degenerate calibration points — homography failed")
        return matrix

    def project(self, x: float, y: float) -> tuple[float, float]:
        """Project one image point (px) to court coordinates (feet)."""

        matrix = self._homography()
        vec = matrix @ np.array([x, y, 1.0])
        if abs(vec[2]) < 1e-12:
            raise ValueError("point projects to infinity")
        return (float(vec[0] / vec[2]), float(vec[1] / vec[2]))

    def to_json(self, path: Path) -> None:
        path.write_text(
            json.dumps(
                {
                    "image_points": [list(p) for p in self.image_points],
                    "court_points": [list(p) for p in self.court_points],
                },
                indent=2,
            )
        )

    @classmethod
    def from_json(cls, path: Path) -> CourtCalibration:
        data = json.loads(Path(path).read_text())
        return cls(
            image_points=tuple(tuple(p) for p in data["image_points"]),
            court_points=tuple(tuple(p) for p in data["court_points"]),
        )
