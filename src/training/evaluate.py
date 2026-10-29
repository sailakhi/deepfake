"""Evaluate the trained deepfake detector on the untouched test split."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.models.baseline import load_model_checkpoint
from src.training.dataset import FaceCropDataset, collate_valid


def evaluate(
    faces_index: Path,
    checkpoint_path: Path,
    batch_size: int,
    image_size: int,
    device_name: str,
) -> None:
    device = torch.device(device_name)

    faces = pd.read_csv(faces_index)
    test_records = faces.loc[faces["split"] == "test"].copy()

    if test_records.empty:
        raise ValueError("Test split contains no face crops.")

    loader = DataLoader(
        FaceCropDataset(
            test_records,
            image_size=image_size,
            augment=False,
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_valid,
    )

    model, checkpoint = load_model_checkpoint(
        checkpoint_path,
        device,
    )

    total = 0
    correct = 0
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    for batch in loader:
        if batch is None:
            continue

        inputs, labels, _ = batch
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            logits = model(inputs)
            predictions = logits.argmax(dim=1)

        total += labels.size(0)
        correct += int((predictions == labels).sum())

        true_positive += int(((predictions == 1) & (labels == 1)).sum())
        true_negative += int(((predictions == 0) & (labels == 0)).sum())
        false_positive += int(((predictions == 1) & (labels == 0)).sum())
        false_negative += int(((predictions == 0) & (labels == 1)).sum())

    if total == 0:
        raise RuntimeError("No readable test face crops were available.")

    accuracy = correct / total

    precision = (
        true_positive / (true_positive + false_positive)
        if (true_positive + false_positive)
        else 0.0
    )

    recall = (
        true_positive / (true_positive + false_negative)
        if (true_positive + false_negative)
        else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )

    print()
    print("===== TEST RESULTS =====")
    print(f"Checkpoint epoch: {checkpoint.get('epoch')}")
    print(f"Test samples:     {total}")
    print(f"Correct:          {correct}")
    print(f"Accuracy:         {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Precision:        {precision:.4f} ({precision * 100:.2f}%)")
    print(f"Recall:           {recall:.4f} ({recall * 100:.2f}%)")
    print(f"F1-score:         {f1:.4f} ({f1 * 100:.2f}%)")
    print()
    print("===== CONFUSION MATRIX =====")
    print("                 Predicted REAL   Predicted FAKE")
    print(f"Actual REAL      {true_negative:14d}   {false_positive:14d}")
    print(f"Actual FAKE      {false_negative:14d}   {true_positive:14d}")
    print()
    print("Actual test distribution:")
    print(test_records["class_name"].value_counts().to_string())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the trained deepfake detector."
    )

    parser.add_argument(
        "--faces-index",
        type=Path,
        default=Path("data/processed/FFPP_FINAL_faces.csv"),
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/ffpp_final/best_model.pt"),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
    )

    parser.add_argument(
        "--device",
        default="cpu",
    )

    args = parser.parse_args()

    evaluate(
        faces_index=args.faces_index,
        checkpoint_path=args.checkpoint,
        batch_size=args.batch_size,
        image_size=args.image_size,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()