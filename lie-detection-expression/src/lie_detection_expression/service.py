"""High-level service orchestration for inference."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import PipelineConfig
from .data import load_openface_csv
from .features import build_frame_features
from .inference import aggregate_windows
from .model import TrainedModel, load_model


def _align_features(frame_features: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    if not feature_columns:
        return frame_features
    aligned = frame_features.reindex(columns=feature_columns, fill_value=0.0)
    return aligned


def predict_clip_from_openface_csv(
    csv_path: str | Path,
    model: TrainedModel,
    config: PipelineConfig,
) -> dict[str, object]:
    df = load_openface_csv(csv_path, config=config)
    feats = build_frame_features(df)
    feats = _align_features(feats, model.feature_columns)
    return aggregate_windows(feats, model=model, config=config)


def predict_clip_with_model_path(
    csv_path: str | Path,
    model_path: str | Path,
    config: PipelineConfig,
) -> dict[str, object]:
    model = load_model(model_path)
    return predict_clip_from_openface_csv(csv_path=csv_path, model=model, config=config)

