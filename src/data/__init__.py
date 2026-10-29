"""Dataset discovery, metadata, and leakage-safe splitting."""

from .discovery import MediaSample, discover_media
from .splits import build_split_report, split_samples

__all__ = ["MediaSample", "build_split_report", "discover_media", "split_samples"]
