"""Interactive hoop selection fallback."""

from __future__ import annotations

import cv2
import numpy as np


def select_hoop_bbox_interactive(
    frame: np.ndarray,
    window_title: str = "Select hoop (draw bbox, press ENTER)",
) -> tuple[float, float, float, float] | None:
    """Prompt the user to draw a hoop bbox on the provided frame."""

    roi = cv2.selectROI(window_title, frame, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(window_title)
    x, y, width, height = (float(value) for value in roi)
    if width <= 0 or height <= 0:
        return None
    return x, y, width, height
