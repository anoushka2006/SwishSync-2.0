"""YOLOv8 object detection adapter."""

from __future__ import annotations

from typing import Any

import numpy as np

from swishsync_cv.config import DetectionConfig
from swishsync_cv.data import DetectionCategory, DetectionRecord


def _normalize_class_name(class_name: str) -> str:
    return class_name.strip().lower().replace("_", " ")


def class_name_to_category(
    class_name: str,
    basketball_aliases: tuple[str, ...],
    hoop_aliases: tuple[str, ...],
) -> DetectionCategory | None:
    """Map a model class name to a SwishSync detection category."""

    normalized = _normalize_class_name(class_name)
    basketball_names = {_normalize_class_name(name) for name in basketball_aliases}
    hoop_names = {_normalize_class_name(name) for name in hoop_aliases}

    if normalized in basketball_names:
        return "basketball"
    if normalized in hoop_names:
        return "hoop"
    return None


def detections_from_yolo_result(
    result: Any,
    frame_index: int,
    timestamp_ms: float,
    config: DetectionConfig,
) -> list[DetectionRecord]:
    """Convert one Ultralytics result object into stable detection records."""

    if result.boxes is None:
        return []

    boxes_xyxy = result.boxes.xyxy.cpu().numpy()
    confidences = result.boxes.conf.cpu().numpy()
    class_ids = result.boxes.cls.cpu().numpy().astype(int)
    names = result.names

    records: list[DetectionRecord] = []
    for bbox, confidence, class_id in zip(boxes_xyxy, confidences, class_ids):
        class_name = names[int(class_id)]
        category = class_name_to_category(
            class_name=class_name,
            basketball_aliases=config.basketball_aliases,
            hoop_aliases=config.hoop_aliases,
        )
        if category is None:
            continue

        records.append(
            DetectionRecord(
                frame_index=frame_index,
                timestamp_ms=timestamp_ms,
                label=category,
                class_name=class_name,
                confidence=float(confidence),
                bbox_xyxy=tuple(float(value) for value in np.asarray(bbox).tolist()),
            )
        )
    return records


class YoloObjectDetector:
    """CPU-only YOLOv8 detector for basketball and hoop classes."""

    def __init__(self, config: DetectionConfig) -> None:
        self.config = config
        self.model = self._load_model(config.model_path)

    @staticmethod
    def _load_model(model_path: str):  # type: ignore[no-untyped-def]
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "Ultralytics is required for YOLOv8 detection. "
                "Install project dependencies with `pip install -e .`."
            ) from exc

        return YOLO(model_path)

    def detect(
        self,
        frame: np.ndarray,
        frame_index: int,
        timestamp_ms: float,
    ) -> list[DetectionRecord]:
        """Run YOLOv8 on one frame and return normalized detections."""

        results = self.model.predict(
            source=frame,
            conf=self.config.confidence_threshold,
            iou=self.config.iou_threshold,
            device=self.config.device,
            verbose=False,
        )
        if not results:
            return []
        return detections_from_yolo_result(
            result=results[0],
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
            config=self.config,
        )
