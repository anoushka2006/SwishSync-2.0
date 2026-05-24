import csv
import json

from swishsync_cv.data import DetectionRecord, FrameDetections, TrajectoryPoint
from swishsync_cv.utils.serialization import (
    write_detections_csv,
    write_detections_jsonl,
    write_trajectory_json,
)


def test_detection_and_trajectory_exports(tmp_path):
    detection = DetectionRecord(
        frame_index=2,
        timestamp_ms=66.6,
        label="basketball",
        class_name="sports ball",
        confidence=0.77,
        bbox_xyxy=(1, 2, 11, 22),
    )
    frame_records = [
        FrameDetections(frame_index=2, timestamp_ms=66.6, detections=[detection])
    ]
    point = TrajectoryPoint(frame_index=2, timestamp_ms=66.6, x=6.0, y=12.0, confidence=0.77)

    jsonl_path = tmp_path / "detections.jsonl"
    csv_path = tmp_path / "detections.csv"
    trajectory_path = tmp_path / "trajectory.json"

    write_detections_jsonl(jsonl_path, frame_records)
    write_detections_csv(csv_path, [detection])
    write_trajectory_json(trajectory_path, [point])

    jsonl_payload = json.loads(jsonl_path.read_text(encoding="utf-8").strip())
    assert jsonl_payload["frame_index"] == 2
    assert jsonl_payload["detections"][0]["center_x"] == 6.0

    with csv_path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert rows[0]["label"] == "basketball"

    trajectory_payload = json.loads(trajectory_path.read_text(encoding="utf-8"))
    assert trajectory_payload[0]["x"] == 6.0
