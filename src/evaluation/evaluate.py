"""Evaluate a trained checkpoint exclusively on the held-out test split."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.evaluation.metrics import calculate_binary_metrics, save_confusion_matrix, save_roc_and_pr_curves, save_training_curves
from src.models.baseline import CLASS_NAMES, load_model_checkpoint
from src.training.dataset import FaceCropDataset, collate_valid
from src.utils import environment_metadata, resolve_device, write_json


def collect_predictions(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> pd.DataFrame:
    """Run inference without gradients and retain per-face held-out predictions."""
    rows: list[dict[str, object]] = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            if batch is None:
                continue
            inputs, labels, sample_ids = batch
            probabilities = torch.softmax(model(inputs.to(device)), dim=1).cpu().numpy()
            for sample_id, label, pair in zip(sample_ids, labels.tolist(), probabilities, strict=True):
                fake_probability = float(pair[1])
                prediction = int(fake_probability >= 0.5)
                rows.append({"sample_id": sample_id, "true_label": int(label), "true_class": CLASS_NAMES[int(label)], "fake_probability": fake_probability, "real_probability": float(pair[0]), "predicted_label": prediction, "predicted_class": CLASS_NAMES[prediction]})
    if not rows:
        raise RuntimeError("No readable held-out test face crops were available for evaluation")
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on only the held-out test face crops.")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best_model.pt"))
    parser.add_argument("--faces-index", type=Path, default=Path("data/processed/faces.csv"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    device = resolve_device(args.device)
    checkpoint_model, checkpoint = load_model_checkpoint(args.checkpoint, device)
    faces = pd.read_csv(args.faces_index)
    test_faces = faces.loc[faces["split"] == "test"]
    if test_faces.empty:
        raise ValueError("No held-out test face crops exist in the face index")
    loader = DataLoader(FaceCropDataset(test_faces, image_size=int(checkpoint.get("image_size", 224))), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_valid)
    predictions = collect_predictions(checkpoint_model, loader, device)
    metrics, matrix = calculate_binary_metrics(predictions["true_label"].to_numpy(), predictions["fake_probability"].to_numpy())
    metrics_dir, plots_dir, confusion_dir, predictions_dir = (args.results_dir / name for name in ("metrics", "plots", "confusion_matrix", "predictions"))
    for directory in (metrics_dir, plots_dir, confusion_dir, predictions_dir):
        directory.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(predictions_dir / "test_predictions.csv", index=False)
    metrics.update({"checkpoint": str(args.checkpoint), "evaluation_split": "test", "environment": environment_metadata(device)})
    write_json(metrics_dir / "test_metrics.json", metrics)
    save_confusion_matrix(matrix, confusion_dir / "test_confusion_matrix.png")
    curves_saved = save_roc_and_pr_curves(predictions["true_label"].to_numpy(), predictions["fake_probability"].to_numpy(), plots_dir)
    training_curves_saved = save_training_curves(checkpoint.get("history", []), plots_dir)
    print("Held-out test metrics:")
    for name in ("accuracy", "precision", "recall", "f1_score", "roc_auc"):
        print(f"{name}: {metrics[name]}")
    print(f"Confusion matrix: {confusion_dir / 'test_confusion_matrix.png'}")
    print(f"ROC/PR curves generated: {curves_saved}")
    print(f"Training curves generated from checkpoint history: {training_curves_saved}")


if __name__ == "__main__":
    main()
