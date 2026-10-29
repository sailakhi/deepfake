"""Deterministic, source-grouped dataset splitting."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SPLITS = ("train", "validation", "test")


def _validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> np.ndarray:
    ratios = np.array([train_ratio, val_ratio, test_ratio], dtype=float)
    if np.any(ratios < 0) or not np.isclose(ratios.sum(), 1.0):
        raise ValueError("train, validation, and test ratios must be non-negative and sum to 1.0")
    return ratios


def _allocation(total: int, ratios: np.ndarray) -> np.ndarray:
    """Allocate integer groups using the largest-remainder method."""
    raw = total * ratios
    counts = np.floor(raw).astype(int)
    for index in np.argsort(-(raw - counts), kind="stable")[: total - counts.sum()]:
        counts[index] += 1
    return counts


def split_samples(
    metadata: pd.DataFrame,
    *,
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> pd.DataFrame:
    """Assign each original-media ``source_id`` to exactly one split.

    Splitting is stratified by class and grouped by ``source_id``. If a source
    has more than one row, all its rows receive the same split. A source mapped
    to inconsistent labels is rejected because that would make stratification
    ambiguous and could conceal a data-quality issue.
    """
    required = {"source_id", "label", "class_name"}
    missing = required.difference(metadata.columns)
    if missing:
        raise ValueError(f"Metadata is missing required columns: {sorted(missing)}")
    ratios = _validate_ratios(train_ratio, val_ratio, test_ratio)
    grouped = metadata.groupby("source_id", sort=True, dropna=False)
    label_counts = grouped["label"].nunique()
    if (label_counts > 1).any():
        bad_sources = label_counts[label_counts > 1].index.tolist()
        raise ValueError(f"A source_id must have one class label: {bad_sources[:3]}")

    group_table = grouped.agg(label=("label", "first"), class_name=("class_name", "first")).reset_index()
    rng = np.random.default_rng(seed)
    assignments: dict[str, str] = {}
    for _, class_groups in group_table.groupby("label", sort=True):
        source_ids = class_groups["source_id"].to_numpy(copy=True)
        rng.shuffle(source_ids)
        counts = _allocation(len(source_ids), ratios)
        start = 0
        for split_name, count in zip(SPLITS, counts, strict=True):
            for source_id in source_ids[start : start + count]:
                assignments[str(source_id)] = split_name
            start += count

    result = metadata.copy()
    result["split"] = result["source_id"].map(assignments)
    if result["split"].isna().any():
        raise RuntimeError("Split assignment failed for one or more records")
    return result.sort_values(["split", "class_name", "sample_id"], kind="stable").reset_index(drop=True)


def assert_no_source_leakage(metadata: pd.DataFrame) -> None:
    """Raise if any source is assigned to more than one split."""
    assignments = metadata.groupby("source_id", dropna=False)["split"].nunique()
    leaking = assignments[assignments > 1]
    if not leaking.empty:
        raise ValueError(f"Source leakage detected for: {leaking.index.tolist()[:3]}")


def build_split_report(metadata: pd.DataFrame) -> pd.DataFrame:
    """Return actual class counts by split, including zero-count combinations."""
    required = {"split", "class_name", "sample_id"}
    missing = required.difference(metadata.columns)
    if missing:
        raise ValueError(f"Cannot build split report; missing columns: {sorted(missing)}")
    table = (
        metadata.groupby(["split", "class_name"], observed=False)["sample_id"]
        .count()
        .unstack(fill_value=0)
        .reindex(index=SPLITS, fill_value=0)
    )
    for class_name in ("real", "fake"):
        if class_name not in table.columns:
            table[class_name] = 0
    table = table[["real", "fake"]]
    table["total"] = table.sum(axis=1)
    return table.rename_axis("split").reset_index()


def write_split_artifacts(metadata: pd.DataFrame, metadata_path: str | Path, splits_dir: str | Path) -> pd.DataFrame:
    """Persist metadata, individual split files, and a truthful count report."""
    assert_no_source_leakage(metadata)
    metadata_target = Path(metadata_path)
    output_dir = Path(splits_dir)
    metadata_target.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(metadata_target, index=False)
    for split_name in SPLITS:
        metadata.loc[metadata["split"] == split_name].to_csv(output_dir / f"{split_name}.csv", index=False)
    report = build_split_report(metadata)
    report.to_csv(output_dir / "split_report.csv", index=False)
    return report
