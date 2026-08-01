import numpy as np

from swishsync_cv.config import VideoOutputConfig
from swishsync_cv.data import DetectionRecord, HoopLock, SparseBallDetection
from swishsync_cv.visualization import render_debug_panel


def test_render_debug_panel_draws_on_copy():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    detections = [
        DetectionRecord(0, 0.0, "basketball", "sports ball", 0.91, (10, 10, 30, 30)),
        DetectionRecord(0, 0.0, "hoop", "rim", 0.88, (50, 20, 80, 50)),
    ]
    hoop_lock = HoopLock(
        center_x=65.0,
        center_y=35.0,
        bbox_xyxy=(50, 20, 80, 50),
        confidence=0.88,
        locked_at_frame=0,
        is_locked=True,
    )
    sparse_point = SparseBallDetection(0, 0.0, 20.0, 20.0, 0.91)

    panel = render_debug_panel(
        frame=frame,
        detections=detections,
        hoop_lock=hoop_lock,
        sparse_point=sparse_point,
        config=VideoOutputConfig(),
        frame_index=0,
        detection_ran=True,
        lifecycle_state="idle",
        candidate_point_count=0,
    )

    assert panel.shape == frame.shape
    assert np.count_nonzero(panel) > 0
    assert np.count_nonzero(frame) == 0
