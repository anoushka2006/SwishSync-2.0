import numpy as np

from swishsync_cv.config import VideoOutputConfig
from swishsync_cv.data import DetectionRecord, TrajectoryPoint
from swishsync_cv.visualization import annotate_frame


def test_annotate_frame_draws_on_copy():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    detections = [
        DetectionRecord(0, 0.0, "basketball", "sports ball", 0.91, (10, 10, 30, 30)),
        DetectionRecord(0, 0.0, "hoop", "rim", 0.88, (50, 20, 80, 50)),
    ]
    trajectory = [
        TrajectoryPoint(0, 0.0, 20, 20, 0.91),
        TrajectoryPoint(1, 33.3, 25, 25, 0.93),
    ]

    annotated = annotate_frame(
        frame=frame,
        detections=detections,
        trajectory=trajectory,
        config=VideoOutputConfig(),
        frame_index=0,
    )

    assert annotated.shape == frame.shape
    assert np.count_nonzero(annotated) > 0
    assert np.count_nonzero(frame) == 0
