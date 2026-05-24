"""OpenCV video reading and writing primitives."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import cv2

from swishsync_cv.data import FramePacket, VideoMetadata


class VideoReader:
    """Context-managed OpenCV video reader that yields ``FramePacket`` objects."""

    def __init__(self, video_path: Path | str) -> None:
        self.video_path = Path(video_path)
        self._capture: cv2.VideoCapture | None = None
        self.metadata: VideoMetadata | None = None

    def __enter__(self) -> "VideoReader":
        if not self.video_path.exists():
            raise FileNotFoundError(f"Input video does not exist: {self.video_path}")

        capture = cv2.VideoCapture(str(self.video_path))
        if not capture.isOpened():
            raise ValueError(f"OpenCV could not open video: {self.video_path}")

        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        if fps <= 0:
            raise ValueError(f"Video FPS must be positive, got {fps}: {self.video_path}")
        if width <= 0 or height <= 0:
            raise ValueError(
                f"Video frame dimensions must be positive, got {width}x{height}: "
                f"{self.video_path}"
            )

        self._capture = capture
        self.metadata = VideoMetadata(
            path=self.video_path,
            fps=fps,
            width=width,
            height=height,
            frame_count=frame_count,
        )
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:  # type: ignore[no-untyped-def]
        if self._capture is not None:
            self._capture.release()
        self._capture = None

    def __iter__(self) -> Iterator[FramePacket]:
        if self._capture is None or self.metadata is None:
            raise RuntimeError("VideoReader must be used as a context manager.")

        frame_index = 0
        while True:
            ok, frame = self._capture.read()
            if not ok:
                break
            timestamp_ms = (frame_index / self.metadata.fps) * 1000.0
            yield FramePacket(index=frame_index, timestamp_ms=timestamp_ms, image=frame)
            frame_index += 1


class VideoWriter:
    """Small wrapper around OpenCV's ``VideoWriter`` with validation."""

    def __init__(
        self,
        output_path: Path | str,
        fps: float,
        frame_size: tuple[int, int],
        codec: str = "mp4v",
    ) -> None:
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.frame_size = frame_size
        self.codec = codec
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self._writer = cv2.VideoWriter(str(self.output_path), fourcc, fps, frame_size)
        if not self._writer.isOpened():
            raise ValueError(f"OpenCV could not create output video: {self.output_path}")

    def write(self, frame) -> None:  # type: ignore[no-untyped-def]
        height, width = frame.shape[:2]
        expected_width, expected_height = self.frame_size
        if (width, height) != (expected_width, expected_height):
            raise ValueError(
                "Output frame size mismatch: "
                f"expected {expected_width}x{expected_height}, got {width}x{height}"
            )
        self._writer.write(frame)

    def release(self) -> None:
        self._writer.release()

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:  # type: ignore[no-untyped-def]
        self.release()
