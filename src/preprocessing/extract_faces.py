"""Extract face crops only after source-level split assignment."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2
import pandas as pd

from .face import FacePreprocessor
from .video import iter_sampled_frames

LOGGER = logging.getLogger(__name__)


def _safe_stem(sample_id: str) -> str:
    return sample_id.replace("/", "__").replace("\\", "__").rsplit(".", 1)[0]


def _write_face(face_bgr, target: Path) -> bool:
    target.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(target), face_bgr))


def extract_faces(
    metadata: pd.DataFrame,
    *,
    output_dir: str | Path,
    image_size: int = 224,
    every_n_frames: int = 15,
    max_frames_per_video: int = 64,
) -> pd.DataFrame:
    """Create face crops whose split is inherited from their source media."""
    required = {"sample_id", "source_id", "class_name", "label", "path", "media_type", "split"}
    missing = required.difference(metadata.columns)
    if missing:
        raise ValueError(f"Metadata is missing required columns: {sorted(missing)}")
    root = Path(output_dir)
    processor = FacePreprocessor(image_size=image_size)
    records: list[dict[str, object]] = []
    for item in metadata.itertuples(index=False):
        base = _safe_stem(str(item.sample_id))
        if item.media_type == "image":
            frame_iter = [(0, None)]
        elif item.media_type == "video":
            try:
                frame_iter = iter_sampled_frames(
                    item.path, every_n_frames=every_n_frames, max_frames=max_frames_per_video
                )
            except ValueError as error:
                LOGGER.warning("Skipping unreadable video %s: %s", item.path, error)
                continue
        else:
            LOGGER.warning("Skipping unsupported media type %s", item.media_type)
            continue
        try:
            for frame_index, frame in frame_iter:
                preparation = processor.prepare_path(item.path) if frame is None else processor.prepare_bgr(frame)
                if preparation.face_bgr is None:
                    continue
                target = root / str(item.split) / str(item.class_name) / f"{base}_frame{frame_index:06d}.jpg"
                if not _write_face(preparation.face_bgr, target):
                    LOGGER.warning("Failed to write face crop: %s", target)
                    continue
                records.append(
                    {
                        "face_path": str(target.resolve()),
                        "sample_id": item.sample_id,
                        "source_id": item.source_id,
                        "class_name": item.class_name,
                        "label": item.label,
                        "split": item.split,
                        "frame_index": frame_index,
                        "face_count": preparation.face_count,
                    }
                )
        except (cv2.error, OSError, ValueError) as error:
            LOGGER.warning("Skipping invalid media %s: %s", item.path, error)
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract leakage-safe face crops after source splitting.")
    parser.add_argument("--metadata", type=Path, default=Path("data/metadata/samples.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/faces"))
    parser.add_argument("--index-path", type=Path, default=Path("data/processed/faces.csv"))
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--every-n-frames", type=int, default=15)
    parser.add_argument("--max-frames-per-video", type=int, default=64)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    faces = extract_faces(
        pd.read_csv(args.metadata),
        output_dir=args.output_dir,
        image_size=args.image_size,
        every_n_frames=args.every_n_frames,
        max_frames_per_video=args.max_frames_per_video,
    )
    args.index_path.parent.mkdir(parents=True, exist_ok=True)
    faces.to_csv(args.index_path, index=False)
    print(f"Extracted {len(faces)} usable face crops to {args.output_dir}")
    print(f"Face index: {args.index_path}")


if __name__ == "__main__":
    main()
