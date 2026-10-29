"""Command-line training for the MobileNetV3-Small baseline."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from src.models.baseline import MODEL_NAME, build_baseline_model, model_summary
from src.training.dataset import FaceCropDataset, collate_valid
from src.utils import environment_metadata, resolve_device, set_seed, write_json


@dataclass
class EpochMetrics:
    loss: float
    accuracy: float
    samples: int


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> EpochMetrics:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    samples = 0
    for batch in loader:
        if batch is None:
            continue
        inputs, labels, _ = batch
        inputs, labels = inputs.to(device), labels.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, labels)
        if training:
            loss.backward()
            optimizer.step()
        batch_size = labels.size(0)
        total_loss += float(loss.detach()) * batch_size
        correct += int((logits.argmax(dim=1) == labels).sum())
        samples += batch_size
    if samples == 0:
        raise RuntimeError("No readable face crops were available for this epoch")
    return EpochMetrics(loss=total_loss / samples, accuracy=correct / samples, samples=samples)


def _loader(records: pd.DataFrame, *, batch_size: int, image_size: int, augment: bool, shuffle: bool, workers: int) -> DataLoader:
    if records.empty:
        raise ValueError("A required split contains no extracted face crops")
    return DataLoader(
        FaceCropDataset(records, image_size=image_size, augment=augment),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        collate_fn=collate_valid,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the Objective 1 MobileNetV3-Small baseline.")
    parser.add_argument("--faces-index", type=Path, default=Path("data/processed/faces.csv"))
    parser.add_argument("--checkpoints-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/training"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-pretrained", action="store_true", help="Do not initialize from ImageNet weights.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.patience <= 0:
        raise ValueError("epochs, batch-size, and patience must be positive")
    set_seed(args.seed)
    device = resolve_device(args.device)
    faces = pd.read_csv(args.faces_index)
    train_records = faces.loc[faces["split"] == "train"]
    validation_records = faces.loc[faces["split"] == "validation"]
    train_loader = _loader(train_records, batch_size=args.batch_size, image_size=args.image_size, augment=True, shuffle=True, workers=args.num_workers)
    validation_loader = _loader(validation_records, batch_size=args.batch_size, image_size=args.image_size, augment=False, shuffle=False, workers=args.num_workers)

    model = build_baseline_model(pretrained=not args.no_pretrained).to(device)
    summary = model_summary(model)
    print(f"Device: {device}")
    print(f"Model: {summary['model_name']}")
    print(f"Parameters: {summary['parameters']:,}; trainable: {summary['trainable_parameters']:,}")
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)
    args.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    history: list[dict[str, float | int]] = []
    best_loss = float("inf")
    stale_epochs = 0
    for epoch in range(1, args.epochs + 1):
        train_metrics = _run_epoch(model, train_loader, criterion, device, optimizer)
        validation_metrics = _run_epoch(model, validation_loader, criterion, device)
        scheduler.step(validation_metrics.loss)
        row = {"epoch": epoch, "train_loss": train_metrics.loss, "train_accuracy": train_metrics.accuracy, "validation_loss": validation_metrics.loss, "validation_accuracy": validation_metrics.accuracy, "learning_rate": optimizer.param_groups[0]["lr"]}
        history.append(row)
        print(f"Epoch {epoch}/{args.epochs} | train loss={train_metrics.loss:.4f} acc={train_metrics.accuracy:.4f} | val loss={validation_metrics.loss:.4f} acc={validation_metrics.accuracy:.4f}")
        checkpoint = {
            "model_name": MODEL_NAME,
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "image_size": args.image_size,
            "pretrained_initialization": not args.no_pretrained,
            "history": history,
            "training_arguments": vars(args),
            "environment": environment_metadata(device),
        }
        if validation_metrics.loss < best_loss:
            best_loss = validation_metrics.loss
            stale_epochs = 0
            torch.save(checkpoint, args.checkpoints_dir / "best_model.pt")
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"Early stopping after {epoch} epochs.")
                break
    torch.save(checkpoint, args.checkpoints_dir / "final_model.pt")
    pd.DataFrame(history).to_csv(args.results_dir / "history.csv", index=False)
    write_json(args.results_dir / "run_metadata.json", {"model": summary, "arguments": vars(args), "environment": environment_metadata(device), "epochs_completed": len(history)})
    print(f"Best checkpoint: {args.checkpoints_dir / 'best_model.pt'}")
    print(f"Final checkpoint: {args.checkpoints_dir / 'final_model.pt'}")


if __name__ == "__main__":
    main()
