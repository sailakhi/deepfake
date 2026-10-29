"""Robust Haar-cascade face extraction and model tensor preparation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
import torch

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class BoundingBox:
    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True)
class FacePreparation:
    """Outcome of attempting to make an image model-ready."""

    tensor: torch.Tensor | None
    face_bgr: np.ndarray | None
    face_count: int
    bbox: BoundingBox | None
    reason: str | None = None


class FaceDetector(Protocol):
    def detect(self, image_bgr: np.ndarray) -> list[BoundingBox]: ...


class HaarCascadeFaceDetector:
    """OpenCV's bundled frontal-face detector; no extra model download needed."""

    def __init__(self, *, scale_factor: float = 1.1, min_neighbors: int = 5, min_size: int = 30) -> None:
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(str(cascade_path))
        if self._cascade.empty():
            raise RuntimeError(f"Unable to load Haar face cascade: {cascade_path}")
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size
    def detect(self, image_bgr: np.ndarray) -> list[BoundingBox]:
        if image_bgr is None or image_bgr.size == 0:
            return []

        max_dimension = 1200
        height, width = image_bgr.shape[:2]
        scale = min(1.0, max_dimension / max(height, width))

        detection_image = image_bgr

        if scale < 1.0:
            detection_image = cv2.resize(
                image_bgr,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_AREA,
            )

        grayscale = cv2.cvtColor(detection_image, cv2.COLOR_BGR2GRAY)

        faces = self._cascade.detectMultiScale(
            grayscale,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=(self.min_size, self.min_size),
        )

        if scale < 1.0:
            inverse_scale = 1.0 / scale
            faces = [
                (
                    int(x * inverse_scale),
                    int(y * inverse_scale),
                    int(w * inverse_scale),
                    int(h * inverse_scale),
                )
                for x, y, w, h in faces
            ]

        return [
            BoundingBox(int(x), int(y), int(w), int(h))
            for x, y, w, h in faces
            ]




def read_bgr(path: str | Path) -> np.ndarray | None:
    """Read a color image without raising on a corrupt or unsupported file."""
    try:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    except cv2.error:
        return None
    return image if image is not None and image.size else None


def largest_box(boxes: list[BoundingBox]) -> BoundingBox | None:
    """Choose the largest detected face, a deterministic multi-face policy."""
    return max(boxes, key=lambda box: (box.width * box.height, -box.y, -box.x), default=None)


def crop_with_margin(image_bgr: np.ndarray, box: BoundingBox, margin: float) -> np.ndarray | None:
    """Crop a bounded face region, expanding it by a proportional margin."""
    height, width = image_bgr.shape[:2]
    extra_x = int(round(box.width * margin))
    extra_y = int(round(box.height * margin))
    left, top = max(0, box.x - extra_x), max(0, box.y - extra_y)
    right, bottom = min(width, box.x + box.width + extra_x), min(height, box.y + box.height + extra_y)
    if right <= left or bottom <= top:
        return None
    crop = image_bgr[top:bottom, left:right]
    return crop if crop.size else None


def face_to_tensor(face_bgr: np.ndarray, *, image_size: int, augment: bool = False) -> torch.Tensor:
    """Resize and normalize an already-cropped BGR face for MobileNetV3."""
    resized = cv2.resize(face_bgr, (image_size, image_size), interpolation=cv2.INTER_AREA)
    if augment and bool(torch.rand(()) < 0.5): 
        resized = cv2.flip(resized, 1)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float().div(255.0)
    mean = torch.tensor(IMAGENET_MEAN, dtype=tensor.dtype).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=tensor.dtype).view(3, 1, 1)
    return (tensor - mean) / std


class FacePreprocessor:
    """Detect, crop, resize, and ImageNet-normalize a face for MobileNetV3."""

    def __init__(self, image_size: int = 224, margin: float = 0.20, detector: FaceDetector | None = None) -> None:
        if image_size <= 0:
            raise ValueError("image_size must be positive")
        if margin < 0:
            raise ValueError("margin must be non-negative")
        self.image_size = image_size
        self.margin = margin
        self.detector = detector or HaarCascadeFaceDetector()

    def prepare_path(self, path: str | Path, *, augment: bool = False) -> FacePreparation:
        image = read_bgr(path)
        if image is None:
            return FacePreparation(None, None, 0, None, "Unreadable or corrupted image")
        return self.prepare_bgr(image, augment=augment)

    def prepare_bgr(self, image_bgr: np.ndarray, *, augment: bool = False) -> FacePreparation:
        if image_bgr is None or image_bgr.size == 0:
            return FacePreparation(None, None, 0, None, "Empty image")
        try:
            boxes = self.detector.detect(image_bgr)
        except cv2.error as error:
            return FacePreparation(None, None, 0, None, f"Face detection failed: {error}")
        box = largest_box(boxes)
        if box is None:
            return FacePreparation(None, None, 0, None, "No face detected")
        face = crop_with_margin(image_bgr, box, self.margin)
        if face is None:
            return FacePreparation(None, None, len(boxes), box, "Invalid face crop")
        return FacePreparation(
            tensor=self.face_to_tensor(face, augment=augment),
            face_bgr=face,
            face_count=len(boxes),
            bbox=box,
        )

    def face_to_tensor(self, face_bgr: np.ndarray, *, augment: bool = False) -> torch.Tensor:
        """Resize and normalize a cropped BGR face; augmentation is train-only."""
        return face_to_tensor(face_bgr, image_size=self.image_size, augment=augment)
