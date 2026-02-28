"""Training entry points."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import PipelineConfig
from .data import load_openface_csv
from .features import build_frame_features
from .model import TrainedModel, save_model, train_model


def train_from_labeled_csv(
    csv_path: str | Path,
    config: PipelineConfig,
    label_column: str = "label",
) -> TrainedModel:
    """
    Train model from frame-level OpenFace CSV with a `label` column.
    Label is expected as 0/1.
    """
    df = load_openface_csv(csv_path, config)
    if label_column not in df.columns:
        raise ValueError(f"Missing label column: {label_column}")

    feats = build_frame_features(df)
    x = feats.to_numpy(dtype=float)
    y = df[label_column].to_numpy(dtype=int)

    model = train_model(x, y, config)
    model.feature_columns = list(feats.columns)
    return model


def train_and_save(
    csv_path: str | Path,
    model_out_path: str | Path,
    config: PipelineConfig,
    label_column: str = "label",
) -> TrainedModel:
    model = train_from_labeled_csv(csv_path, config=config, label_column=label_column)
    save_model(model, model_out_path)
    return model

