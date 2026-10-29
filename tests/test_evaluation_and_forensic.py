from pathlib import Path

import numpy as np

from src.evaluation.metrics import calculate_binary_metrics, save_confusion_matrix
from src.forensic.result import build_forensic_result, save_forensic_result


def test_binary_metrics_are_calculated_from_predictions() -> None:
    metrics, matrix = calculate_binary_metrics(np.array([0, 0, 1, 1]), np.array([0.1, 0.7, 0.8, 0.9]))
    assert metrics["accuracy"] == 0.75
    assert metrics["roc_auc"] == 1.0
    assert matrix.tolist() == [[1, 1], [0, 2]]


def test_one_class_metrics_do_not_fabricate_roc_auc() -> None:
    metrics, _ = calculate_binary_metrics(np.array([0, 0]), np.array([0.1, 0.2]))
    assert metrics["roc_auc"] is None


def test_forensic_schema_and_json_output(tmp_path: Path) -> None:
    result = build_forensic_result(
        input_path="example.jpg",
        media_type="image",
        model_name="mobilenet_v3_small",
        prediction="UNKNOWN",
        fake_probability=None,
        confidence=None,
        face_detection={"faces_detected": 0},
        reason="No face detected",
    )
    output = tmp_path / "forensic.json"
    save_forensic_result(result, output)
    assert output.exists()
    assert result["prediction"] == "UNKNOWN"
    assert result["confidence"] is None


def test_confusion_plot_is_written(tmp_path: Path) -> None:
    output = tmp_path / "matrix.png"
    save_confusion_matrix(np.array([[2, 1], [0, 3]]), output)
    assert output.exists() and output.stat().st_size > 0
