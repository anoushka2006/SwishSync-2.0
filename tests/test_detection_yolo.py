import numpy as np

from swishsync_cv.config import DetectionConfig
from swishsync_cv.detection.yolo import (
    class_name_to_category,
    detections_from_yolo_result,
)


class FakeTensor:
    def __init__(self, values):
        self.values = np.asarray(values)

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class FakeBoxes:
    def __init__(self):
        self.xyxy = FakeTensor([[1, 2, 11, 22], [30, 40, 50, 60], [0, 0, 5, 5]])
        self.conf = FakeTensor([0.9, 0.8, 0.7])
        self.cls = FakeTensor([0, 1, 2])


class FakeResult:
    def __init__(self):
        self.boxes = FakeBoxes()
        self.names = {0: "sports ball", 1: "rim", 2: "person"}


def test_class_name_to_category_maps_aliases():
    config = DetectionConfig()

    assert (
        class_name_to_category(
            "sports_ball",
            config.basketball_aliases,
            config.hoop_aliases,
        )
        == "basketball"
    )
    assert (
        class_name_to_category("Basketball Rim", config.basketball_aliases, config.hoop_aliases)
        == "hoop"
    )
    assert class_name_to_category("person", config.basketball_aliases, config.hoop_aliases) is None


def test_detections_from_yolo_result_filters_to_supported_categories():
    records = detections_from_yolo_result(
        result=FakeResult(),
        frame_index=4,
        timestamp_ms=133.3,
        config=DetectionConfig(),
    )

    assert [record.label for record in records] == ["basketball", "hoop"]
    assert records[0].center == (6.0, 12.0)
    assert records[1].bbox_xyxy == (30.0, 40.0, 50.0, 60.0)
