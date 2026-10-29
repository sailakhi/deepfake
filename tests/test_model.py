from pathlib import Path

import torch

from src.models.baseline import MODEL_NAME, build_baseline_model, load_model_checkpoint, model_summary


def test_baseline_output_shape_and_binary_logits() -> None:
    model = build_baseline_model(pretrained=False).eval()
    with torch.inference_mode():
        output = model(torch.zeros((2, 3, 64, 64)))
    assert output.shape == (2, 2)
    summary = model_summary(model)
    assert summary["model_name"] == MODEL_NAME
    assert summary["parameters"] == summary["trainable_parameters"]


def test_model_checkpoint_loads(tmp_path: Path) -> None:
    original = build_baseline_model(pretrained=False)
    checkpoint_path = tmp_path / "model.pt"
    torch.save({"model_name": MODEL_NAME, "model_state_dict": original.state_dict(), "image_size": 224}, checkpoint_path)

    loaded, metadata = load_model_checkpoint(checkpoint_path, torch.device("cpu"))

    assert metadata["model_name"] == MODEL_NAME
    assert not loaded.training
