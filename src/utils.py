"""Shared reproducibility, device, and serialization helpers."""

from __future__ import annotations

import json
import platform
import random
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def set_seed(seed: int) -> None:
    """Set Python, NumPy, and PyTorch seeds for repeatable experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(requested: str = "auto") -> torch.device:
    """Select CUDA only when available, or validate an explicit device."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in this PyTorch installation")
    return device


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def read_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def environment_metadata(device: torch.device) -> dict[str, str | bool | None]:
    return {
        "timestamp_utc": utc_now(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
    }
