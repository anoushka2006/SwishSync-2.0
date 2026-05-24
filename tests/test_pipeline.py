import cv2
import numpy as np

from swishsync_cv.config import DetectionConfig, PipelineConfig, VideoOutputConfig
from swishsync_cv.data import DetectionRecord
from swishsync_cv.pipeline import run_pipeline


class FakeDetector:
    def detect(self, frame, frame_index: int, timestamp_ms: float):
        x = 10 + frame_index * 5
        return [
            DetectionRecord(
                frame_index=frame_index,
                timestamp_ms=timestamp_ms,
                label="basketball",
                class_name="sports ball",
                confidence=0.9,
                bbox_xyxy=(x, 20, x + 10, 30),
            ),
            DetectionRecord(
                frame_index=frame_index,
                timestamp_ms=timestamp_ms,
                label="hoop",
                class_name="rim",
                confidence=0.8,
                bbox_xyxy=(50, 10, 80, 35),
            ),
        ]


def _write_test_video(path, frame_count: int = 3) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (96, 64),
    )
    assert writer.isOpened()
    for index in range(frame_count):
        frame = np.zeros((64, 96, 3), dtype=np.uint8)
        frame[:, :, 1] = index * 40
        writer.write(frame)
    writer.release()


def test_run_pipeline_with_fake_detector_exports_artifacts(tmp_path):
    input_video = tmp_path / "input.mp4"
    _write_test_video(input_video)

    config = PipelineConfig(
        input_video=input_video,
        output_dir=tmp_path / "outputs",
        detection=DetectionConfig(confidence_threshold=0.25),
        video_output=VideoOutputConfig(save_debug_frames=True, debug_frame_stride=1),
    )

    result = run_pipeline(config=config, detector=FakeDetector())

    assert result.processed_frames == 3
    assert result.detection_count == 6
    assert result.trajectory_point_count == 3
    assert result.output_video_path.exists()
    assert result.detections_jsonl_path.exists()
    assert result.detections_csv_path.exists()
    assert result.trajectory_json_path.exists()
    assert len(list(config.debug_frames_dir.glob("*.jpg"))) == 3
