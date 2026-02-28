"""Training and model persistence."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from .config import PipelineConfig


@dataclass(slots=True)
class TrainedModel:
    pipeline: Pipeline
    feature_columns: list[str]

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self.pipeline.predict_proba(x)


def train_model(x_train: np.ndarray, y_train: np.ndarray, config: PipelineConfig) -> TrainedModel:
    """Train baseline ANN-like classifier using sklearn MLP."""
    clf = MLPClassifier(
        hidden_layer_sizes=config.hidden_layers,
        activation="relu",
        solver="adam",
        learning_rate_init=config.learning_rate,
        batch_size=config.batch_size,
        max_iter=config.max_epochs,
        random_state=42,
    )
    pipeline = Pipeline(
        [
            ("scaler", MinMaxScaler()),
            ("clf", clf),
        ]
    )
    pipeline.fit(x_train, y_train)
    return TrainedModel(pipeline=pipeline, feature_columns=[])


def save_model(model: TrainedModel, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: str | Path) -> TrainedModel:
    return joblib.load(path)

