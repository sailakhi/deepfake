"""Datasets for already-extracted face crops."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from src.preprocessing.face import face_to_tensor, read_bgr


class FaceCropDataset(Dataset[tuple[torch.Tensor, int, str] | None]):
    """Read processed face crops; invalid entries become ``None`` safely."""

    def __init__(self, records: pd.DataFrame, *, image_size: int = 224, augment: bool = False) -> None:
        required = {"face_path", "label", "sample_id"}
        missing = required.difference(records.columns)
        if missing:
            raise ValueError(f"Face index is missing columns: {sorted(missing)}")
        self.records = records.reset_index(drop=True)
        self.augment = augment
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str] | None:
        record = self.records.iloc[index]
        image = read_bgr(Path(record.face_path))
        if image is None:
            return None
        try:
            tensor = face_to_tensor(image, image_size=self.image_size, augment=self.augment)
        except Exception:
            return None
        return tensor, int(record.label), str(record.sample_id)


def collate_valid(batch: list[tuple[torch.Tensor, int, str] | None]) -> tuple[torch.Tensor, torch.Tensor, list[str]] | None:
    """Drop individual corrupt face files without terminating an epoch."""
    valid = [item for item in batch if item is not None]
    if not valid:
        return None
    tensors, labels, sample_ids = zip(*valid, strict=True)
    return torch.stack(list(tensors)), torch.tensor(labels, dtype=torch.long), list(sample_ids)
