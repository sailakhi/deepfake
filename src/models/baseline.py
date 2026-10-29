"""MobileNetV3-Small baseline for binary facial deepfake classification."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

CLASS_NAMES = {0: "REAL", 1: "FAKE"}
MODEL_NAME = "mobilenet_v3_small"


def build_baseline_model(*, pretrained: bool = True) -> nn.Module:
    """Build MobileNetV3-Small with a two-logit classifier head.

    This compact transfer-learning architecture is appropriate for CPU
    demonstrations and small controlled subsets. ``pretrained=True`` obtains
    official ImageNet initialization through TorchVision when not already
    cached; it does not claim deepfake-specific pretraining.
    """
    weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = mobilenet_v3_small(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, 2)
    return model


def model_summary(model: nn.Module) -> dict[str, int | str]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {"model_name": MODEL_NAME, "parameters": total, "trainable_parameters": trainable}


def load_model_checkpoint(path: str | Path, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    """Load a saved baseline checkpoint for evaluation or inference."""
    checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
    if checkpoint.get("model_name") != MODEL_NAME:
        raise ValueError(f"Unsupported checkpoint model: {checkpoint.get('model_name')}")
    model = build_baseline_model(pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model, checkpoint
