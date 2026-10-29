from pathlib import Path

import cv2
import numpy as np
import torch

from src.preprocessing.face import BoundingBox, FacePreprocessor


class FixedDetector:
    def detect(self, image_bgr: np.ndarray) -> list[BoundingBox]:
        return [BoundingBox(10, 10, 40, 30), BoundingBox(1, 1, 10, 10)]


class EmptyDetector:
    def detect(self, image_bgr: np.ndarray) -> list[BoundingBox]:
        return []


def test_face_preparation_resizes_normalizes_and_selects_largest_face() -> None:
    processor = FacePreprocessor(image_size=32, margin=0.0, detector=FixedDetector())
    image = np.full((80, 90, 3), 127, dtype=np.uint8)

    result = processor.prepare_bgr(image)

    assert result.tensor is not None
    assert result.tensor.shape == (3, 32, 32)
    assert result.tensor.dtype == torch.float32
    assert result.face_count == 2
    assert result.bbox == BoundingBox(10, 10, 40, 30)


def test_no_face_is_unknown_ready_result() -> None:
    result = FacePreprocessor(detector=EmptyDetector()).prepare_bgr(np.zeros((20, 20, 3), dtype=np.uint8))
    assert result.tensor is None
    assert result.reason == "No face detected"


def test_corrupted_image_path_does_not_raise(tmp_path: Path) -> None:
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not an image")
    result = FacePreprocessor(detector=EmptyDetector()).prepare_path(bad)
    assert result.tensor is None
    assert result.reason == "Unreadable or corrupted image"


def test_face_tensor_is_deterministic_without_augmentation() -> None:
    processor = FacePreprocessor(image_size=16, detector=FixedDetector())
    image = np.full((80, 90, 3), 127, dtype=np.uint8)
    first = processor.prepare_bgr(image).tensor
    second = processor.prepare_bgr(image).tensor
    assert first is not None and second is not None
    assert torch.equal(first, second)
