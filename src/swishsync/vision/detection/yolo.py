"""YoloDetector — platform Detector wrapping the legacy YoloObjectDetector.

Phase-A adapter: it does NOT reimplement inference. It builds the existing
``swishsync_cv`` YOLO detector from a ``DetectionConfig`` (device supplied by the
backend) and maps its ``DetectionRecord`` output into IR ``Detection``. The
underlying legacy detector is exposed as ``.legacy`` so the wholesale
``LegacyShotEngine`` can drive its stride/detect_crop paths directly (the legacy
engine works in DetectionRecord space, not IR).
"""

from __future__ import annotations

from swishsync.core.registry import register_detector
from swishsync.core.schemas import Detection, ObjectClass
from swishsync.vision.backends import InferenceBackend
from swishsync.vision.interfaces import Detector

from swishsync_cv.config import DetectionConfig
from swishsync_cv.detection.yolo import YoloObjectDetector

# Legacy category label -> IR ObjectClass. The legacy detector only ever emits
# "basketball" / "hoop" (see class_name_to_category); anything else is dropped.
_LABEL_TO_CLASS = {
    "basketball": ObjectClass.BALL,
    "hoop": ObjectClass.HOOP,
}


@register_detector("yolo")
class YoloDetector(Detector):
    """frame pixels -> IR Detections via the proven legacy YOLO adapter."""

    def __init__(
        self,
        backend: InferenceBackend,
        weights: str,
        confidence: float = 0.25,
    ) -> None:
        super().__init__(backend)
        self.legacy = YoloObjectDetector(
            DetectionConfig(
                model_path=weights,
                confidence_threshold=confidence,
                device=backend.device,
            )
        )

    def detect(self, frame, frame_index: int, t_ms: float) -> list[Detection]:
        detections: list[Detection] = []
        for record in self.legacy.detect(
            frame=frame,
            frame_index=frame_index,
            timestamp_ms=t_ms,
        ):
            cls = _LABEL_TO_CLASS.get(record.label)
            if cls is None:
                continue
            detections.append(
                Detection(
                    frame=frame_index,
                    t_ms=t_ms,
                    cls=cls,
                    bbox_xyxy=record.bbox_xyxy,
                    conf=record.confidence,
                )
            )
        return detections
