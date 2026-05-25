"""Detection modules for basketball and hoop localization."""

from swishsync_cv.detection.hoop_detector import HybridHoopDetector, HoopCandidate
from swishsync_cv.detection.yolo import YoloObjectDetector, class_name_to_category

__all__ = [
    "HybridHoopDetector",
    "HoopCandidate",
    "YoloObjectDetector",
    "class_name_to_category",
]
