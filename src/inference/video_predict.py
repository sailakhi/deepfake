"""Sample video frames, classify usable faces, and aggregate transparently."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from src.forensic.result import build_forensic_result, save_forensic_result
from src.inference.predictor import BinaryPrediction, predict_tensor
from src.models.baseline import MODEL_NAME, load_model_checkpoint
from src.preprocessing.face import FacePreprocessor
from src.preprocessing.video import inspect_video, iter_sampled_frames
from src.utils import resolve_device


def aggregate_video_predictions(predictions: list[BinaryPrediction]) -> dict[str, float | int | str] | None:
    """Aggregate video evidence as the arithmetic mean of valid-frame P(FAKE).

    ``mean_fake_probability = sum(P(FAKE)_i) / n``. The final class is FAKE
    when that mean is at least 0.5. Confidence is ``max(mean, 1 - mean)`` and,
    like individual probabilities, is not calibrated.
    """
    if not predictions:
        return None
    average_fake = sum(item.fake_probability for item in predictions) / len(predictions)
    label = int(average_fake >= 0.5)
    prediction = "FAKE" if label else "REAL"
    return {
        "prediction": prediction,
        "predicted_label": label,
        "fake_probability": average_fake,
        "real_probability": 1.0 - average_fake,
        "confidence": max(average_fake, 1.0 - average_fake),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict facial deepfake class for a video.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best_model.pt"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--every-n-frames", type=int, default=15)
    parser.add_argument("--max-frames", type=int, default=64)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    device = resolve_device(args.device)
    model, checkpoint = load_model_checkpoint(args.checkpoint, device)
    video_info = inspect_video(args.input)
    preprocessor = FacePreprocessor(image_size=int(checkpoint.get("image_size", 224)))
    frame_rows: list[dict[str, Any]] = []
    predictions: list[BinaryPrediction] = []
    sampled_frames = 0
    no_face_frames = 0
    for frame_index, frame in iter_sampled_frames(args.input, every_n_frames=args.every_n_frames, max_frames=args.max_frames):
        sampled_frames += 1
        preparation = preprocessor.prepare_bgr(frame)
        row: dict[str, Any] = {"frame_index": frame_index, "faces_detected": preparation.face_count, "status": "no_face"}
        if preparation.tensor is None:
            no_face_frames += 1
            row["reason"] = preparation.reason
        else:
            prediction = predict_tensor(model, preparation.tensor, device)
            predictions.append(prediction)
            row.update({"status": "analyzed", "prediction": prediction.prediction, "real_probability": prediction.real_probability, "fake_probability": prediction.fake_probability, "confidence": prediction.confidence})
        frame_rows.append(row)
    aggregate = aggregate_video_predictions(predictions)
    real_frames = sum(item.prediction == "REAL" for item in predictions)
    fake_frames = sum(item.prediction == "FAKE" for item in predictions)
    summary = {
        "total_frames": video_info.total_frames,
        "fps": video_info.fps,
        "sampled_frames": sampled_frames,
        "frames_analyzed": len(predictions),
        "frames_with_no_face": no_face_frames,
        "real_frame_count": real_frames,
        "fake_frame_count": fake_frames,
        "aggregation": "mean_fake_probability",
    }
    output_dir = args.results_dir / "forensic"
    predictions_dir = args.results_dir / "predictions"
    frame_csv = predictions_dir / f"{args.input.stem}_frame_predictions.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(frame_rows).to_csv(frame_csv, index=False)
    if aggregate is None:
        result = build_forensic_result(input_path=args.input, media_type="video", model_name=MODEL_NAME, prediction="UNKNOWN", fake_probability=None, confidence=None, face_detection={"analyzed_frames": 0, "no_face_frames": no_face_frames}, frame_summary=summary, frame_predictions=frame_rows, reason="No usable face was detected in sampled frames")
        print(f"Video: {args.input.name}\nPrediction: UNKNOWN\nReason: No usable face was detected")
    else:
        result = build_forensic_result(input_path=args.input, media_type="video", model_name=MODEL_NAME, prediction=str(aggregate["prediction"]), fake_probability=float(aggregate["fake_probability"]), confidence=float(aggregate["confidence"]), face_detection={"analyzed_frames": len(predictions), "no_face_frames": no_face_frames}, frame_summary=summary, frame_predictions=frame_rows)
        print(f"Video: {args.input.name}\nTotal frames: {video_info.total_frames}\nSampled frames: {sampled_frames}\nSuccessfully analyzed frames: {len(predictions)}\nFrames with no face: {no_face_frames}\nREAL frame count: {real_frames}\nFAKE frame count: {fake_frames}\nAverage P(FAKE): {float(aggregate['fake_probability']):.2%}\nFinal prediction: {aggregate['prediction']}\nConfidence: {float(aggregate['confidence']):.2%}")
    report_path = output_dir / f"{args.input.stem}_video_result.json"
    save_forensic_result(result, report_path)
    print(f"Frame-level CSV: {frame_csv}\nForensic report: {report_path}")


if __name__ == "__main__":
    main()
