"""Shared binary prediction helpers for model-ready face tensors."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from src.models.baseline import CLASS_NAMES


@dataclass(frozen=True)
class BinaryPrediction:
    prediction: str
    predicted_label: int
    real_probability: float
    fake_probability: float
    confidence: float


def predict_tensor(model: torch.nn.Module, tensor: torch.Tensor, device: torch.device) -> BinaryPrediction:
    """Classify one normalized face tensor with raw, uncalibrated probabilities."""
    with torch.inference_mode():
        probabilities = torch.softmax(model(tensor.unsqueeze(0).to(device)), dim=1)[0].detach().cpu()
    real_probability, fake_probability = float(probabilities[0]), float(probabilities[1])
    label = int(fake_probability >= real_probability)
    return BinaryPrediction(
        prediction=CLASS_NAMES[label],
        predicted_label=label,
        real_probability=real_probability,
        fake_probability=fake_probability,
        confidence=max(real_probability, fake_probability),
    )
