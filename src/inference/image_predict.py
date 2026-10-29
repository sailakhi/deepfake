"""Predict a REAL/FAKE label for one image when a usable face is present."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.forensic.result import build_forensic_result, save_forensic_result
from src.inference.predictor import predict_tensor
from src.models.baseline import MODEL_NAME, load_model_checkpoint
from src.preprocessing.face import FacePreprocessor
from src.utils import resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict facial deepfake class for an image.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best_model.pt"))
    parser.add_argument("--output", type=Path, default=None, help="Forensic JSON output path.")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    device = resolve_device(args.device)
    model, checkpoint = load_model_checkpoint(args.checkpoint, device)
    preprocessor = FacePreprocessor(image_size=int(checkpoint.get("image_size", 224)))
    preparation = preprocessor.prepare_path(args.input)
    output = args.output or Path("results/forensic") / f"{args.input.stem}_image_result.json"
    face_info = {
        "faces_detected": preparation.face_count,
        "selected_bbox": vars(preparation.bbox) if preparation.bbox else None,
    }
    if preparation.tensor is None:
        result = build_forensic_result(
            input_path=args.input,
            media_type="image",
            model_name=MODEL_NAME,
            prediction="UNKNOWN",
            fake_probability=None,
            confidence=None,
            face_detection=face_info,
            reason=preparation.reason,
        )
        print(f"Input: {args.input.name}\nPrediction: UNKNOWN\nReason: {preparation.reason}")
    else:
        prediction = predict_tensor(model, preparation.tensor, device)
        result = build_forensic_result(
            input_path=args.input,
            media_type="image",
            model_name=MODEL_NAME,
            prediction=prediction.prediction,
            fake_probability=prediction.fake_probability,
            confidence=prediction.confidence,
            face_detection=face_info,
        )
        print(f"Input: {args.input.name}\nPrediction: {prediction.prediction}\nConfidence: {prediction.confidence:.2%}\nP(FAKE): {prediction.fake_probability:.2%}")
    save_forensic_result(result, output)
    print(f"Forensic report: {output}")


if __name__ == "__main__":
    main()
