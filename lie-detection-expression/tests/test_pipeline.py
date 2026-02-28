from __future__ import annotations

import numpy as np
import pandas as pd

from lie_detection_expression.config import PipelineConfig
from lie_detection_expression.data import load_openface_csv
from lie_detection_expression.features import build_frame_features
from lie_detection_expression.inference import aggregate_windows
from lie_detection_expression.model import load_model, save_model, train_model
from lie_detection_expression.openface import _locate_openface_csv, resolve_openface_binary
from lie_detection_expression.sentence_scoring import _score_sentence


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "frame": [1, 2, 3, 4],
            "timestamp": [0.0, 0.03, 0.06, 0.09],
            "confidence": [0.9, 0.8, 0.6, 0.95],
            "success": [1, 1, 1, 0],
            "AU01_c": [1, 0, 1, 1],
            "AU01_r": [2.0, 1.0, 0.5, 3.0],
            "AU02_c": [0, 1, 1, 1],
            "AU02_r": [0.2, 1.2, 0.8, 2.5],
            "label": [1, 0, 1, 0],
        }
    )


def test_openface_filtering(tmp_path):
    csv_path = tmp_path / "sample.csv"
    _sample_df().to_csv(csv_path, index=False)

    config = PipelineConfig(min_confidence=0.7, require_success=True)
    out = load_openface_csv(csv_path, config=config)
    assert len(out) == 2


def test_openface_filtering_with_spaced_headers(tmp_path):
    csv_path = tmp_path / "spaced.csv"
    csv_path.write_text(
        "frame, timestamp, confidence, success, AU01_c, AU01_r\n"
        "1, 0.0, 0.9, 1, 1, 2.0\n"
        "2, 0.1, 0.6, 1, 0, 1.0\n",
        encoding="utf-8",
    )
    out = load_openface_csv(csv_path, config=PipelineConfig(min_confidence=0.7, require_success=True))
    assert len(out) == 1


def test_feature_builder():
    feats = build_frame_features(_sample_df())
    assert "AU01_pxi" in feats.columns
    assert "AU02_pxi" in feats.columns
    assert len(feats) == 4


def test_train_save_load_and_infer(tmp_path):
    df = _sample_df()
    feats = build_frame_features(df)
    x = feats.to_numpy(dtype=float)
    y = df["label"].to_numpy(dtype=int)

    model = train_model(x, y, config=PipelineConfig(max_epochs=20))
    model.feature_columns = list(feats.columns)

    model_path = tmp_path / "model.joblib"
    save_model(model, model_path)
    loaded = load_model(model_path)

    result = aggregate_windows(
        frame_features=feats,
        model=loaded,
        config=PipelineConfig(fps=2, window_seconds=1.0),
    )
    assert 0.0 <= result["clip_probability"] <= 1.0
    assert len(result["window_probabilities"]) > 0


def test_sentence_window_alignment_weighted_mean():
    windows = [
        {"start": 0.0, "end": 1.0, "probability": 0.2},
        {"start": 1.0, "end": 2.0, "probability": 0.8},
    ]
    p = _score_sentence(0.5, 1.5, windows)
    assert abs(p - 0.5) < 1e-9


def test_locate_openface_csv_prefers_named_file(tmp_path):
    expected = tmp_path / "video1.csv"
    fallback = tmp_path / "other.csv"
    fallback.write_text("a,b\n1,2\n", encoding="utf-8")
    expected.write_text("a,b\n3,4\n", encoding="utf-8")
    picked = _locate_openface_csv(tmp_path, "video1")
    assert picked == expected


def test_resolve_openface_binary_from_openface_dir(monkeypatch, tmp_path):
    fake = tmp_path / "FeatureExtraction"
    fake.write_text("", encoding="utf-8")
    monkeypatch.delenv("OPENFACE_BIN", raising=False)
    monkeypatch.setenv("OPENFACE_DIR", str(tmp_path))
    picked = resolve_openface_binary()
    assert picked == str(fake)
