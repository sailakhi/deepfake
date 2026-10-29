from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.training.dataset import FaceCropDataset, collate_valid
from src.training.train import _run_epoch


class TinyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.classifier = nn.Linear(3, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(inputs.mean(dim=(2, 3)))


def test_face_crop_dataset_skips_corrupt_file_and_training_epoch_runs(tmp_path: Path) -> None:
    image_path = tmp_path / "face.jpg"
    assert cv2.imwrite(str(image_path), np.full((24, 24, 3), 128, dtype=np.uint8))
    bad_path = tmp_path / "bad.jpg"
    bad_path.write_bytes(b"corrupt")
    records = pd.DataFrame(
        [
            {"face_path": str(image_path), "label": 0, "sample_id": "good"},
            {"face_path": str(bad_path), "label": 1, "sample_id": "bad"},
        ]
    )
    loader = DataLoader(FaceCropDataset(records, image_size=16), batch_size=2, collate_fn=collate_valid)
    model = TinyClassifier()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    metrics = _run_epoch(model, loader, nn.CrossEntropyLoss(), torch.device("cpu"), optimizer)
    assert metrics.samples == 1
    assert 0.0 <= metrics.accuracy <= 1.0
