"""Forensic result schema shared by image and video inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.utils import utc_now, write_json


def build_forensic_result(
    *,
    input_path: str | Path,
    media_type: str,
    model_name: str,
    prediction: str,
    fake_probability: float | None,
    confidence: float | None,
    face_detection: dict[str, Any],
    frame_summary: dict[str, Any] | None = None,
    frame_predictions: list[dict[str, Any]] | None = None,
    reason: str | None = None,
) -> dict[str, Any]:
    """Build a transparent report; probabilities are never called calibrated."""
    return {
        "schema_version": "1.0",
        "analysis_timestamp_utc": utc_now(),
        "input_filename": Path(input_path).name,
        "input_path": str(input_path),
        "media_type": media_type,
        "model": {"name": model_name, "prediction_labels": {"0": "REAL", "1": "FAKE"}},
        "prediction": prediction,
        "fake_probability": fake_probability,
        "confidence": confidence,
        "confidence_definition": "max(P(REAL), P(FAKE)); raw model softmax probability, not calibrated" if confidence is not None else None,
        "face_detection": face_detection,
        "frame_summary": frame_summary,
        "frame_predictions": frame_predictions or [],
        "reason": reason,
    }


def save_forensic_result(result: dict[str, Any], path: str | Path) -> None:
    write_json(path, result)
