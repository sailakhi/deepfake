"""Command-line dataset discovery and leakage-safe splitting."""

from __future__ import annotations

import argparse
from pathlib import Path

from .discovery import discover_media, samples_to_frame
from .splits import split_samples, write_split_artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create source-level deepfake dataset splits.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--metadata-path", type=Path, default=Path("data/metadata/samples.csv"))
    parser.add_argument("--splits-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    samples = discover_media(args.raw_dir)
    metadata = split_samples(
        samples_to_frame(samples),
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )
    report = write_split_artifacts(metadata, args.metadata_path, args.splits_dir)
    print(f"Discovered {len(metadata)} original media files from {args.raw_dir}")
    print("Source-level split report (frames are not extracted until after this step):")
    print(report.to_string(index=False))
    print(f"Metadata: {args.metadata_path}")


if __name__ == "__main__":
    main()
