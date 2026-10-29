"""Discover labelled source media without extracting frames."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

CLASS_TO_LABEL = {"real": 0, "fake": 1}
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
VIDEO_SUFFIXES = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}


@dataclass(frozen=True)
class MediaSample:
    """A labelled original image or video before any frame extraction."""

    sample_id: str
    source_id: str
    class_name: str
    label: int
    path: str
    media_type: str


def _normalise_class_directory(path: Path) -> str | None:
    """Return a supported class name for a directory name, if any."""
    name = path.name.casefold()
    return name if name in CLASS_TO_LABEL else None


def _media_type(path: Path) -> str | None:
    suffix = path.suffix.casefold()
    if suffix in IMAGE_SUFFIXES:
        return "image"
    if suffix in VIDEO_SUFFIXES:
        return "video"
    return None


def discover_media(raw_dir: str | Path) -> list[MediaSample]:
    """Discover media under ``raw_dir/{real,fake}``.

    Each file is treated as an original media unit. This is intentional: the
    train/validation/test assignment is made here, before a video can be
    converted to frames. ``source_id`` therefore remains stable for every
    extracted face from that media file.
    """
    root = Path(raw_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")

    samples: list[MediaSample] = []
    class_directories = (item for item in root.iterdir() if item.is_dir())
    for class_dir in sorted(
        class_directories,
        key=lambda path: (CLASS_TO_LABEL.get(path.name.casefold(), len(CLASS_TO_LABEL)), path.name.casefold()),
    ):
        class_name = _normalise_class_directory(class_dir)
        if class_name is None:
            continue
        for path in sorted((item for item in class_dir.rglob("*") if item.is_file()), key=lambda p: p.as_posix()):
            media_type = _media_type(path)
            if media_type is None:
                continue
            relative = path.relative_to(root).as_posix()
            # The relative source path includes the class directory so that
            # filenames duplicated across classes cannot be merged by mistake.
            source_id = relative.rsplit(".", 1)[0]
            samples.append(
                MediaSample(
                    sample_id=relative,
                    source_id=source_id,
                    class_name=class_name,
                    label=CLASS_TO_LABEL[class_name],
                    path=str(path),
                    media_type=media_type,
                )
            )
    if not samples:
        raise ValueError(
            f"No supported image or video files found under {root}. "
            "Expected data/raw/real and data/raw/fake."
        )
    return samples


def samples_to_frame(samples: list[MediaSample]) -> pd.DataFrame:
    """Convert discovered media to a deterministic metadata table."""
    frame = pd.DataFrame(asdict(sample) for sample in samples)
    return frame.sort_values(["class_name", "sample_id"], kind="stable").reset_index(drop=True)
