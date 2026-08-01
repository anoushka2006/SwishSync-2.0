"""Single-person pose landmark extractor (ultralytics YOLO-pose, CPU).

Uses ultralytics YOLO-pose instead of mediapipe: mediapipe segfaults on
Python 3.13, and ultralytics is already a core dependency and runs CPU-only
(the no-GPU reason pose was wanted in the first place). COCO's 17 keypoints
are remapped to MediaPipe's index scheme so posture.compute_posture() — which
is pose-source-agnostic — needs no changes.
"""

from __future__ import annotations

import numpy as np

from swishsync_cv.pose.posture import Landmark

# ultralytics/COCO keypoint index -> MediaPipe Pose index used by posture.py
_COCO_TO_MP = {
    5: 11, 6: 12,   # shoulders
    7: 13, 8: 14,   # elbows
    9: 15, 10: 16,  # wrists
    11: 23, 12: 24,  # hips
    13: 25, 14: 26,  # knees
    15: 27, 16: 28,  # ankles
}


class PoseEstimator:
    """Extract shooter pose landmarks from BGR frames via YOLO-pose."""

    def __init__(
        self,
        model_path: str = "yolov8n-pose.pt",
        min_confidence: float = 0.3,
    ) -> None:
        from ultralytics import YOLO

        self._model = YOLO(model_path)
        self._min_confidence = min_confidence

    def landmarks(self, frame_bgr: np.ndarray) -> dict[int, Landmark] | None:
        """Return {MediaPipe index: (x_px, y_px, confidence)} for the top person."""

        results = self._model.predict(frame_bgr, device="cpu", verbose=False)
        if not results:
            return None
        keypoints = results[0].keypoints
        if keypoints is None or len(keypoints) == 0:
            return None
        # highest-scoring detected person
        person = keypoints.data[0]
        landmarks: dict[int, Landmark] = {}
        for coco_index, mp_index in _COCO_TO_MP.items():
            x, y, conf = person[coco_index]
            landmarks[mp_index] = (float(x), float(y), float(conf))
        return landmarks

    def close(self) -> None:  # symmetry with a resource-owning estimator
        pass

    def __enter__(self) -> PoseEstimator:
        return self

    def __exit__(self, *_exc) -> None:
        self.close()
