"""Deterministic, source-grouped dataset splitting."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


SPLITS = ("train", "validation", "test")


def _validate_ratios(
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> np.ndarray:
    ratios = np.array(
        [train_ratio, val_ratio, test_ratio],
        dtype=float,
    )

    if np.any(ratios < 0) or not np.isclose(ratios.sum(), 1.0):
        raise ValueError(
            "train, validation, and test ratios must be non-negative "
            "and sum to 1.0"
        )

    return ratios


def _allocation(total: int, ratios: np.ndarray) -> np.ndarray:
    """Allocate integer groups using the largest-remainder method."""

    raw = total * ratios
    counts = np.floor(raw).astype(int)

    for index in np.argsort(
        -(raw - counts),
        kind="stable",
    )[: total - counts.sum()]:
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
    """Assign each split group to exactly one dataset split.

    If ``split_group_id`` exists, it is used for grouping. This allows
    FaceForensics++ related media to remain in the same split.

    For older metadata without ``split_group_id``, ``source_id`` is used.
    """

    required = {"source_id", "label", "class_name"}

    missing = required.difference(metadata.columns)

    if missing:
        raise ValueError(
            f"Metadata is missing required columns: {sorted(missing)}"
        )

    ratios = _validate_ratios(
        train_ratio,
        val_ratio,
        test_ratio,
    )

    grouping_column = (
        "split_group_id"
        if "split_group_id" in metadata.columns
        else "source_id"
    )

    grouped = metadata.groupby(
        grouping_column,
        sort=True,
        dropna=False,
    )

    label_counts = grouped["label"].nunique()

    if (label_counts > 1).any():
        bad_groups = label_counts[label_counts > 1].index.tolist()

        raise ValueError(
            f"A {grouping_column} must have one class label: "
            f"{bad_groups[:3]}"
        )

    group_table = (
        grouped.agg(
            label=("label", "first"),
            class_name=("class_name", "first"),
        )
        .reset_index()
    )

    rng = np.random.default_rng(seed)

    assignments: dict[str, str] = {}

    for _, class_groups in group_table.groupby(
        "label",
        sort=True,
    ):
        group_ids = class_groups[grouping_column].to_numpy(
            copy=True
        )

        rng.shuffle(group_ids)

        counts = _allocation(
            len(group_ids),
            ratios,
        )

        start = 0

        for split_name, count in zip(
            SPLITS,
            counts,
            strict=True,
        ):
            for group_id in group_ids[start : start + count]:
                assignments[str(group_id)] = split_name

            start += count

    result = metadata.copy()

    result["split"] = result[grouping_column].map(
        lambda value: assignments[str(value)]
    )

    if result["split"].isna().any():
        raise RuntimeError(
            "Split assignment failed for one or more records"
        )

    return result.sort_values(
        ["split", "class_name", "sample_id"],
        kind="stable",
    ).reset_index(drop=True)


def assert_no_source_leakage(
    metadata: pd.DataFrame,
) -> None:
    """Raise if any split group is assigned to multiple splits."""

    grouping_column = (
        "split_group_id"
        if "split_group_id" in metadata.columns
        else "source_id"
    )

    assignments = metadata.groupby(
        grouping_column,
        dropna=False,
    )["split"].nunique()

    leaking = assignments[assignments > 1]

    if not leaking.empty:
        raise ValueError(
            f"Source leakage detected for: "
            f"{leaking.index.tolist()[:3]}"
        )


def build_split_report(
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Return actual class counts by split."""

    required = {
        "split",
        "class_name",
        "sample_id",
    }

    missing = required.difference(metadata.columns)

    if missing:
        raise ValueError(
            "Cannot build split report; "
            f"missing columns: {sorted(missing)}"
        )

    table = (
        metadata.groupby(
            ["split", "class_name"],
            observed=False,
        )["sample_id"]
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


def write_split_artifacts(
    metadata: pd.DataFrame,
    metadata_path: str | Path,
    splits_dir: str | Path,
) -> pd.DataFrame:
    """Persist metadata, split files, and the split report."""

    assert_no_source_leakage(metadata)

    metadata_target = Path(metadata_path)
    output_dir = Path(splits_dir)

    metadata_target.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata.to_csv(
        metadata_target,
        index=False,
    )

    for split_name in SPLITS:
        metadata.loc[
            metadata["split"] == split_name
        ].to_csv(
            output_dir / f"{split_name}.csv",
            index=False,
        )

    report = build_split_report(metadata)

    report.to_csv(
        output_dir / "split_report.csv",
        index=False,
    )

    return report