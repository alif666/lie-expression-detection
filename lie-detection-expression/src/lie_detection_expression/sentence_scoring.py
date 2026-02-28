"""Sentence-level scoring by aligning transcript timings with AU window scores."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import PipelineConfig
from .data import load_openface_csv
from .features import build_frame_features
from .inference import window_probabilities_with_times
from .model import load_model
from .service import _align_features


@dataclass(slots=True)
class SentenceResult:
    text: str
    start: float
    end: float
    deception_probability: float
    level: str


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def _score_sentence(
    sent_start: float,
    sent_end: float,
    windows: list[dict[str, float]],
) -> float:
    weighted_sum = 0.0
    total_weight = 0.0
    for w in windows:
        ov = _overlap(sent_start, sent_end, float(w["start"]), float(w["end"]))
        if ov > 0.0:
            weighted_sum += ov * float(w["probability"])
            total_weight += ov
    if total_weight > 0.0:
        return weighted_sum / total_weight

    # Fallback: nearest window center when there is no overlap.
    sent_center = (sent_start + sent_end) / 2.0
    nearest = min(
        windows,
        key=lambda w: abs(sent_center - ((float(w["start"]) + float(w["end"])) / 2.0)),
    )
    return float(nearest["probability"])


def _level(probability: float) -> str:
    return "higher" if probability >= 0.5 else "lower"


def _transcribe_with_whisper(video_path: str | Path, model_name: str) -> list[dict[str, Any]]:
    try:
        import whisper  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Missing transcription dependency `openai-whisper`. "
            "Install with: python -m pip install openai-whisper"
        ) from exc

    model = whisper.load_model(model_name)
    result = model.transcribe(str(video_path))
    segments = result.get("segments", [])
    normalized: list[dict[str, Any]] = []
    for seg in segments:
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start))
        if end <= start:
            end = start + 0.01
        normalized.append({"text": text, "start": start, "end": end})
    return normalized


def score_video_sentences(
    video_path: str | Path,
    openface_csv_path: str | Path,
    model_path: str | Path,
    config: PipelineConfig,
    whisper_model_name: str = "base",
) -> dict[str, Any]:
    transcripts = _transcribe_with_whisper(video_path=video_path, model_name=whisper_model_name)
    if not transcripts:
        raise ValueError("No transcript segments detected from the video.")

    openface_df = load_openface_csv(openface_csv_path, config=config)
    features = build_frame_features(openface_df)

    model = load_model(model_path)
    features = _align_features(features, model.feature_columns)
    timestamps = openface_df["timestamp"].to_numpy(dtype=float)
    windows = window_probabilities_with_times(
        frame_features=features,
        timestamps=timestamps,
        model=model,
        config=config,
    )

    results: list[SentenceResult] = []
    for item in transcripts:
        p = _score_sentence(float(item["start"]), float(item["end"]), windows)
        results.append(
            SentenceResult(
                text=str(item["text"]),
                start=float(item["start"]),
                end=float(item["end"]),
                deception_probability=float(p),
                level=_level(p),
            )
        )

    clip_prob = float(np.mean([r.deception_probability for r in results]))
    return {
        "video_path": str(video_path),
        "openface_csv_path": str(openface_csv_path),
        "model_path": str(model_path),
        "clip_probability": clip_prob,
        "sentences": [asdict(r) for r in results],
    }


def write_sentence_scores(payload: dict[str, Any], out_json: str | Path, out_csv: str | Path) -> None:
    out_json = Path(out_json)
    out_csv = Path(out_csv)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with out_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["text", "start", "end", "deception_probability", "level"],
        )
        writer.writeheader()
        for row in payload.get("sentences", []):
            writer.writerow(row)
