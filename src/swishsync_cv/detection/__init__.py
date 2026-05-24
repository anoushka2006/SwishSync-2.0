"""Detection modules for basketball and hoop localization."""

from swishsync_cv.detection.yolo import YoloObjectDetector, class_name_to_category

__all__ = ["YoloObjectDetector", "class_name_to_category"]
