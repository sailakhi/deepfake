"""Binary metrics and plots calculated only from supplied predictions."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str((Path.cwd() / "work" / "matplotlib").resolve()))

import matplotlib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    f1_score,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def calculate_binary_metrics(labels: np.ndarray, fake_probabilities: np.ndarray) -> tuple[dict[str, Any], np.ndarray]:
    """Calculate metrics for REAL=0 and FAKE=1, guarding one-class test sets."""
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(fake_probabilities, dtype=float)
    if labels.ndim != 1 or probabilities.ndim != 1 or len(labels) != len(probabilities) or not len(labels):
        raise ValueError("labels and fake_probabilities must be non-empty one-dimensional arrays of equal length")
    predictions = (probabilities >= 0.5).astype(int)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    metrics: dict[str, Any] = {
        "samples": int(len(labels)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1_score": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probabilities)) if len(np.unique(labels)) == 2 else None,
        "positive_class": "FAKE",
        "probability_definition": "raw softmax P(FAKE); not calibrated",
    }
    return metrics, matrix


def save_confusion_matrix(matrix: np.ndarray, output_path: str | Path) -> None:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(5, 4))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["REAL", "FAKE"], yticklabels=["REAL", "FAKE"], ylabel="True label", xlabel="Predicted label", title="Held-out Test Confusion Matrix")
    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(2):
        for column in range(2):
            axis.text(column, row, str(int(matrix[row, column])), ha="center", va="center", color="white" if matrix[row, column] > threshold else "black")
    figure.tight_layout()
    figure.savefig(target, dpi=160)
    plt.close(figure)


def save_roc_and_pr_curves(labels: np.ndarray, probabilities: np.ndarray, output_dir: str | Path) -> bool:
    """Save ROC and PR curves when the held-out labels include both classes."""
    labels, probabilities = np.asarray(labels), np.asarray(probabilities)
    if len(np.unique(labels)) < 2:
        return False
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    fpr, tpr, _ = roc_curve(labels, probabilities)
    precision, recall, _ = precision_recall_curve(labels, probabilities)
    roc_auc = roc_auc_score(labels, probabilities)
    average_precision = average_precision_score(labels, probabilities)
    for filename, x, y, xlabel, ylabel, title, annotation in [
        ("roc_curve.png", fpr, tpr, "False Positive Rate", "True Positive Rate", "ROC Curve", f"AUC = {roc_auc:.3f}"),
        ("precision_recall_curve.png", recall, precision, "Recall", "Precision", "Precision-Recall Curve", f"AP = {average_precision:.3f}"),
    ]:
        figure, axis = plt.subplots(figsize=(5, 4))
        axis.plot(x, y, label=annotation)
        if filename == "roc_curve.png":
            axis.plot([0, 1], [0, 1], "--", color="gray", label="Chance")
        axis.set(xlabel=xlabel, ylabel=ylabel, title=title)
        axis.legend(loc="lower right")
        figure.tight_layout()
        figure.savefig(output / filename, dpi=160)
        plt.close(figure)
    return True


def save_training_curves(history: list[dict[str, Any]], output_dir: str | Path) -> bool:
    """Create actual loss and accuracy plots from checkpointed epoch history."""
    if not history:
        return False
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in history]
    for metric, title, ylabel, filename in [
        ("loss", "Training and Validation Loss", "Cross-Entropy Loss", "loss.png"),
        ("accuracy", "Training and Validation Accuracy", "Accuracy", "accuracy.png"),
    ]:
        figure, axis = plt.subplots(figsize=(6, 4))
        axis.plot(epochs, [row[f"train_{metric}"] for row in history], marker="o", label="Train")
        axis.plot(epochs, [row[f"validation_{metric}"] for row in history], marker="o", label="Validation")
        axis.set(xlabel="Epoch", ylabel=ylabel, title=title)
        axis.legend()
        figure.tight_layout()
        figure.savefig(output / filename, dpi=160)
        plt.close(figure)
    return True
