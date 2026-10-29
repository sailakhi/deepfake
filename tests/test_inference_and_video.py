from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from src.inference.predictor import predict_tensor
from src.inference.video_predict import aggregate_video_predictions
from src.preprocessing.video import inspect_video, iter_sampled_frames


class AlwaysFake(torch.nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.tensor([[0.0, 2.0]], dtype=torch.float32).repeat(inputs.shape[0], 1)


def test_binary_prediction_has_valid_confidence_and_fake_label() -> None:
    prediction = predict_tensor(AlwaysFake(), torch.zeros((3, 16, 16)), torch.device("cpu"))
    assert prediction.prediction == "FAKE"
    assert 0.5 <= prediction.confidence <= 1.0
    assert 0.0 <= prediction.fake_probability <= 1.0


def test_video_mean_probability_aggregation_and_empty_handling() -> None:
    predictions = [
        predict_tensor(AlwaysFake(), torch.zeros((3, 8, 8)), torch.device("cpu")),
        predict_tensor(AlwaysFake(), torch.zeros((3, 8, 8)), torch.device("cpu")),
    ]
    aggregate = aggregate_video_predictions(predictions)
    assert aggregate is not None and aggregate["prediction"] == "FAKE"
    assert aggregate_video_predictions([]) is None


def test_video_frame_sampling(tmp_path: Path) -> None:
    path = tmp_path / "sample.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (32, 32))
    if not writer.isOpened():
        pytest.skip("OpenCV video writer is unavailable in this environment")
    for value in range(5):
        writer.write(np.full((32, 32, 3), value * 30, dtype=np.uint8))
    writer.release()
    info = inspect_video(path)
    frames = list(iter_sampled_frames(path, every_n_frames=2, max_frames=3))
    assert info.total_frames == 5
    assert [index for index, _ in frames] == [0, 2, 4]


def test_invalid_video_is_reported() -> None:
    with pytest.raises(ValueError, match="Unable to open video"):
        inspect_video("missing.mp4")
