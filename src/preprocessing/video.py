"""Safe, bounded frame sampling for videos."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    total_frames: int
    fps: float


def inspect_video(path: str | Path) -> VideoInfo:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"Unable to open video: {path}")
    try:
        return VideoInfo(
            total_frames=max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT))),
            fps=float(capture.get(cv2.CAP_PROP_FPS) or 0.0),
        )
    finally:
        capture.release()


def iter_sampled_frames(
    path: str | Path, *, every_n_frames: int = 15, max_frames: int | None = 64
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield sampled BGR frames; unreadable videos raise a useful ValueError."""
    if every_n_frames <= 0:
        raise ValueError("every_n_frames must be positive")
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be positive when provided")
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"Unable to open video: {path}")
    emitted = 0
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % every_n_frames == 0:
                yield frame_index, frame
                emitted += 1
                if max_frames is not None and emitted >= max_frames:
                    break
            frame_index += 1
    finally:
        capture.release()
