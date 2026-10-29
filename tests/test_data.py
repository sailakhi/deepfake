from pathlib import Path

import pandas as pd
import pytest

from src.data.discovery import discover_media, samples_to_frame
from src.data.splits import assert_no_source_leakage, build_split_report, split_samples


def _write_sample(root: Path, class_name: str, filename: str) -> None:
    target = root / class_name / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"test media")


def test_discover_media_labels_and_types(tmp_path: Path) -> None:
    _write_sample(tmp_path, "REAL", "one.jpg")
    _write_sample(tmp_path, "fake", "nested/two.mp4")
    _write_sample(tmp_path, "fake", "ignore.txt")

    samples = discover_media(tmp_path)

    assert [(item.class_name, item.label, item.media_type) for item in samples] == [
        ("real", 0, "image"),
        ("fake", 1, "video"),
    ]


def test_source_group_split_is_reproducible_and_leakage_safe() -> None:
    rows = []
    for label, class_name in ((0, "real"), (1, "fake")):
        for source in range(10):
            for frame in range(2):
                rows.append(
                    {
                        "sample_id": f"{class_name}-{source}-{frame}",
                        "source_id": f"{class_name}-{source}",
                        "label": label,
                        "class_name": class_name,
                    }
                )
    metadata = pd.DataFrame(rows)
    first = split_samples(metadata, seed=7)
    second = split_samples(metadata, seed=7)

    pd.testing.assert_frame_equal(first, second)
    assert_no_source_leakage(first)
    assert set(first["split"]) == {"train", "validation", "test"}
    assert first.groupby("source_id")["split"].nunique().eq(1).all()


def test_split_rejects_conflicting_source_labels() -> None:
    metadata = pd.DataFrame(
        [
            {"source_id": "same", "label": 0, "class_name": "real"},
            {"source_id": "same", "label": 1, "class_name": "fake"},
        ]
    )
    with pytest.raises(ValueError, match="one class label"):
        split_samples(metadata)


def test_split_report_contains_actual_totals() -> None:
    metadata = pd.DataFrame(
        [
            {"sample_id": "a", "source_id": "a", "label": 0, "class_name": "real", "split": "train"},
            {"sample_id": "b", "source_id": "b", "label": 1, "class_name": "fake", "split": "test"},
        ]
    )
    report = build_split_report(metadata).set_index("split")
    assert report.loc["train", "real"] == 1
    assert report.loc["test", "fake"] == 1
    assert report.loc["validation", "total"] == 0
