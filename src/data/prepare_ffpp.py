"""Prepare a leakage-safe FaceForensics++ Deepfakes dataset."""

from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

import pandas as pd


SPLITS = ("train", "validation", "test")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create leakage-safe FaceForensics++ metadata and splits."
    )

    parser.add_argument(
        "--real-dir",
        type=Path,
        required=True,
        help="FaceForensics++ original video directory.",
    )

    parser.add_argument(
        "--fake-dir",
        type=Path,
        required=True,
        help="FaceForensics++ Deepfakes video directory.",
    )

    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=Path("data/metadata/ffpp_samples.csv"),
    )

    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=Path("data/splits/ffpp"),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
    )

    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
    )

    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
    )

    return parser


def validate_directories(real_dir: Path, fake_dir: Path) -> None:
    if not real_dir.is_dir():
        raise FileNotFoundError(
            f"REAL directory not found: {real_dir}"
        )

    if not fake_dir.is_dir():
        raise FileNotFoundError(
            f"FAKE directory not found: {fake_dir}"
        )


def discover_real_videos(real_dir: Path) -> dict[str, Path]:
    videos = {}

    for path in sorted(real_dir.glob("*.mp4")):
        videos[path.stem] = path

    if not videos:
        raise ValueError(
            f"No REAL MP4 videos found in {real_dir}"
        )

    return videos


def discover_fake_pairs(fake_dir: Path) -> dict[tuple[str, str], list[Path]]:
    pattern = re.compile(r"^(\d+)_(\d+)\.mp4$", re.IGNORECASE)

    groups: dict[tuple[str, str], list[Path]] = {}

    for path in sorted(fake_dir.glob("*.mp4")):
        match = pattern.fullmatch(path.name)

        if match is None:
            raise ValueError(
                f"Unexpected FAKE filename format: {path.name}"
            )

        first, second = match.groups()
        pair = tuple(sorted((first, second)))

        groups.setdefault(pair, []).append(path)

    if not groups:
        raise ValueError(
            f"No valid FAKE pair videos found in {fake_dir}"
        )

    return groups


def create_records(
    real_videos: dict[str, Path],
    fake_groups: dict[tuple[str, str], list[Path]],
) -> tuple[list[dict], list[tuple[str, str]]]:

    records: list[dict] = []
    valid_groups: list[tuple[str, str]] = []

    for pair, fake_paths in sorted(fake_groups.items()):
        first, second = pair

        first_real = real_videos.get(first)
        second_real = real_videos.get(second)

        if first_real is None or second_real is None:
            missing = [
                value
                for value, path in (
                    (first, first_real),
                    (second, second_real),
                )
                if path is None
            ]

            print(
                f"WARNING: skipping pair {first}_{second}; "
                f"missing REAL video(s): {missing}"
            )
            continue

        split_group_id = f"ffpp_pair_{first}_{second}"

        records.append(
            {
                "sample_id": f"real/{first}.mp4",
                "source_id": first,
                "class_name": "real",
                "label": 0,
                "path": str(first_real),
                "media_type": "video",
                "split_group_id": split_group_id,
            }
        )

        records.append(
            {
                "sample_id": f"real/{second}.mp4",
                "source_id": second,
                "class_name": "real",
                "label": 0,
                "path": str(second_real),
                "media_type": "video",
                "split_group_id": split_group_id,
            }
        )

        for fake_path in sorted(fake_paths):
            records.append(
                {
                    "sample_id": f"fake/{fake_path.name}",
                    "source_id": fake_path.stem,
                    "class_name": "fake",
                    "label": 1,
                    "path": str(fake_path),
                    "media_type": "video",
                    "split_group_id": split_group_id,
                }
            )

        valid_groups.append(pair)

    if not valid_groups:
        raise ValueError(
            "No valid FF++ pair groups could be created."
        )

    return records, valid_groups


def assign_splits(
    records: list[dict],
    groups: list[tuple[str, str]],
    *,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> None:

    total_ratio = train_ratio + val_ratio + test_ratio

    if abs(total_ratio - 1.0) > 1e-9:
        raise ValueError(
            "train, validation, and test ratios must sum to 1.0"
        )

    if any(
        ratio < 0
        for ratio in (
            train_ratio,
            val_ratio,
            test_ratio,
        )
    ):
        raise ValueError(
            "Split ratios must be non-negative."
        )

    shuffled_groups = list(groups)

    random.Random(seed).shuffle(shuffled_groups)

    total = len(shuffled_groups)

    train_count = round(total * train_ratio)
    val_count = round(total * val_ratio)

    if train_count + val_count > total:
        val_count = total - train_count

    split_by_group: dict[str, str] = {}

    for index, pair in enumerate(shuffled_groups):
        group_id = f"ffpp_pair_{pair[0]}_{pair[1]}"

        if index < train_count:
            split_name = "train"
        elif index < train_count + val_count:
            split_name = "validation"
        else:
            split_name = "test"

        split_by_group[group_id] = split_name

    for record in records:
        group_id = record["split_group_id"]
        record["split"] = split_by_group[group_id]


def validate_no_leakage(metadata: pd.DataFrame) -> None:
    counts = metadata.groupby(
        "split_group_id"
    )["split"].nunique()

    leaking = counts[counts > 1]

    if not leaking.empty:
        raise RuntimeError(
            "Data leakage detected between splits: "
            f"{leaking.index.tolist()[:5]}"
        )


def write_outputs(
    metadata: pd.DataFrame,
    metadata_path: Path,
    splits_dir: Path,
) -> None:

    metadata_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    splits_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata.to_csv(
        metadata_path,
        index=False,
    )

    for split_name in SPLITS:
        split_data = metadata[
            metadata["split"] == split_name
        ]

        split_data.to_csv(
            splits_dir / f"{split_name}.csv",
            index=False,
        )

    report = (
        metadata.groupby(
            ["split", "class_name"]
        )["sample_id"]
        .count()
        .unstack(fill_value=0)
        .reindex(index=SPLITS, fill_value=0)
    )

    for class_name in ("real", "fake"):
        if class_name not in report.columns:
            report[class_name] = 0

    report = report[["real", "fake"]]
    report["total"] = report["real"] + report["fake"]

    report.to_csv(
        splits_dir / "split_report.csv"
    )

    print()
    print("FF++ split report:")
    print(report)
    print()
    print(f"Metadata: {metadata_path}")
    print(f"Splits:   {splits_dir}")


def main() -> None:
    args = build_parser().parse_args()

    validate_directories(
        args.real_dir,
        args.fake_dir,
    )

    real_videos = discover_real_videos(
        args.real_dir
    )

    fake_groups = discover_fake_pairs(
        args.fake_dir
    )

    records, groups = create_records(
        real_videos,
        fake_groups,
    )

    assign_splits(
        records,
        groups,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    metadata = pd.DataFrame(records)

    validate_no_leakage(metadata)

    metadata = metadata.sort_values(
        ["split", "class_name", "sample_id"],
        kind="stable",
    ).reset_index(drop=True)

    metadata_path = Path(args.metadata_path).resolve()
    splits_dir = Path(args.splits_dir).resolve()

    print(f"WRITING METADATA TO: {metadata_path}")
    print(f"WRITING SPLITS TO: {splits_dir}")

    write_outputs(
        metadata,
        metadata_path,
        splits_dir,
    )
       
    print(
        f"Discovered {len(real_videos)} REAL videos."
    )
    print(
        f"Discovered {sum(len(v) for v in fake_groups.values())} "
        "FAKE videos."
    )
    print(
        f"Created {len(groups)} leakage-safe FF++ pair groups."
    )
    print(
        f"Created {len(metadata)} metadata records."
    )


if __name__ == "__main__":
    main()