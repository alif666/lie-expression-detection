"""FastAPI app for clip-level deception inference."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from lie_detection_expression.config import PipelineConfig
from lie_detection_expression.service import predict_clip_with_model_path


class PredictRequest(BaseModel):
    openface_csv_path: str = Field(..., description="Path to OpenFace frame-level CSV.")


class PredictResponse(BaseModel):
    clip_probability: float
    window_probabilities: list[float]
    window_size_frames: int


app = FastAPI(title="Lie Detection Expression API", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    model_path = os.environ.get("LIE_MODEL_PATH", "models/baseline.joblib")
    csv_path = Path(payload.openface_csv_path)
    if not csv_path.exists():
        raise HTTPException(status_code=400, detail=f"CSV not found: {csv_path}")

    resolved_model = Path(model_path)
    if not resolved_model.exists():
        raise HTTPException(status_code=500, detail=f"Model not found: {resolved_model}")

    try:
        result = predict_clip_with_model_path(
            csv_path=csv_path,
            model_path=resolved_model,
            config=PipelineConfig(),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return PredictResponse(**result)

