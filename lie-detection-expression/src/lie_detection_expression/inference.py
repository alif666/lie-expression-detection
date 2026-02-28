"""Windowed inference utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PipelineConfig
from .model import TrainedModel


def aggregate_windows(
    frame_features: pd.DataFrame,
    model: TrainedModel,
    config: PipelineConfig,
) -> dict[str, object]:
    """
    Predict deception probability per window and aggregate to clip score.
    Uses non-overlapping 2-second windows by default.
    """
    x = frame_features.to_numpy(dtype=float)
    window_size = max(config.window_size_frames, 1)

    if len(x) == 0:
        raise ValueError("No frames available for inference.")

    windows: list[float] = []
    for start in range(0, len(x), window_size):
        chunk = x[start : start + window_size]
        if len(chunk) == 0:
            continue
        chunk_mean = np.mean(chunk, axis=0, keepdims=True)
        prob = float(model.predict_proba(chunk_mean)[0, 1])
        windows.append(prob)

    clip_probability = float(np.mean(windows))
    return {
        "clip_probability": clip_probability,
        "window_probabilities": windows,
        "window_size_frames": window_size,
    }


def window_probabilities_with_times(
    frame_features: pd.DataFrame,
    timestamps: np.ndarray,
    model: TrainedModel,
    config: PipelineConfig,
) -> list[dict[str, float]]:
    """Predict per-window probabilities and return time ranges for alignment."""
    x = frame_features.to_numpy(dtype=float)
    if len(x) == 0:
        raise ValueError("No frames available for inference.")
    if len(timestamps) != len(x):
        raise ValueError("Timestamps length must match number of feature rows.")

    window_size = max(config.window_size_frames, 1)
    out: list[dict[str, float]] = []

    for start in range(0, len(x), window_size):
        end = min(start + window_size, len(x))
        chunk = x[start:end]
        chunk_mean = np.mean(chunk, axis=0, keepdims=True)
        prob = float(model.predict_proba(chunk_mean)[0, 1])
        out.append(
            {
                "start": float(timestamps[start]),
                "end": float(timestamps[end - 1]),
                "probability": prob,
            }
        )
    return out
