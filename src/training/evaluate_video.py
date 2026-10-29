"""Evaluate deepfake predictions at the video level."""

from pathlib import Path
from collections import defaultdict
import argparse

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.models.baseline import load_model_checkpoint
from src.training.dataset import FaceCropDataset, collate_valid


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate FaceForensics++ test videos."
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
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    faces = pd.read_csv(args.faces_index)
    test = faces.loc[faces["split"] == "test"].copy()

    if test.empty:
        raise ValueError("No test face crops were found.")

    # Every video must have a single ground-truth label.
    label_counts = test.groupby("sample_id")["label"].nunique()
    if (label_counts > 1).any():
        raise ValueError("Some videos have inconsistent ground-truth labels.")

    dataset = FaceCropDataset(
        test, image_size=args.image_size, augment=False
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_valid,
    )

    model, checkpoint = load_model_checkpoint(args.checkpoint, device)

    probability_sums = defaultdict(float)
    crop_counts = defaultdict(int)

    model.eval()
    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue

            images, labels, sample_ids = batch
            images = images.to(device)

            logits = model(images)
            fake_probabilities = torch.softmax(logits, dim=1)[:, 1]

            for sample_id, probability in zip(
                sample_ids, fake_probabilities.cpu().tolist()
            ):
                probability_sums[sample_id] += probability
                crop_counts[sample_id] += 1

    # Build one record per video.
    label_lookup = (
        test.drop_duplicates("sample_id")
        .set_index("sample_id")["label"]
        .to_dict()
    )

    rows = []
    for sample_id, probability_sum in probability_sums.items():
        count = crop_counts[sample_id]
        average_fake_probability = probability_sum / count
        actual = int(label_lookup[sample_id])
        predicted = int(average_fake_probability >= 0.5)

        rows.append({
            "sample_id": sample_id,
            "actual_label": actual,
            "actual_class": "FAKE" if actual == 1 else "REAL",
            "average_fake_probability": average_fake_probability,
            "predicted_label": predicted,
            "predicted_class": "FAKE" if predicted == 1 else "REAL",
            "face_crops_used": count,
        })

    results = pd.DataFrame(rows).sort_values("sample_id")
    if results.empty:
        raise RuntimeError("No readable test face crops were evaluated.")

    actual = results["actual_label"]
    predicted = results["predicted_label"]

    tp = int(((actual == 1) & (predicted == 1)).sum())
    tn = int(((actual == 0) & (predicted == 0)).sum())
    fp = int(((actual == 0) & (predicted == 1)).sum())
    fn = int(((actual == 1) & (predicted == 0)).sum())

    accuracy = (tp + tn) / len(results)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    output_path = Path("results/ffpp_final/video_predictions.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)

    print("\n===== VIDEO-LEVEL TEST RESULTS =====")
    print(f"Checkpoint epoch: {checkpoint.get('epoch')}")
    print(f"Videos evaluated: {len(results)}")
    print(f"Correct predictions: {tp + tn}")
    print(f"Accuracy:  {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Precision: {precision:.4f} ({precision * 100:.2f}%)")
    print(f"Recall:    {recall:.4f} ({recall * 100:.2f}%)")
    print(f"F1-score:  {f1:.4f} ({f1 * 100:.2f}%)")
    print("\n===== CONFUSION MATRIX =====")
    print("                 Predicted REAL   Predicted FAKE")
    print(f"Actual REAL      {tn:14d}   {fp:14d}")
    print(f"Actual FAKE      {fn:14d}   {tp:14d}")
    print("\nVideos evaluated by actual class:")
    print(results["actual_class"].value_counts().to_string())
    print(f"\nPer-video predictions saved to: {output_path}")


if __name__ == "__main__":
    main()