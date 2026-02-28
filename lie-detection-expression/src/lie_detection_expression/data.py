"""Data loading utilities for OpenFace output."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import PipelineConfig


REQUIRED_BASE_COLUMNS = {"frame", "timestamp", "confidence", "success"}


def load_openface_csv(path: str | Path, config: PipelineConfig) -> pd.DataFrame:
    """Load and filter OpenFace frame-level data."""
    df = pd.read_csv(path)
    # OpenFace CSV headers often include spaces after commas.
    df.columns = [str(c).strip() for c in df.columns]
    missing = REQUIRED_BASE_COLUMNS - set(df.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"Missing required columns: {missing_list}")

    mask = df["confidence"] >= config.min_confidence
    if config.require_success:
        mask = mask & (df["success"] == 1)
    return df.loc[mask].reset_index(drop=True)
