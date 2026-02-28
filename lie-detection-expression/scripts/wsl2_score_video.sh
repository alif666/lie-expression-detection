#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <video_path> [model_path]"
  echo "Example: $0 sample_data/alif_sample_1.mp4 models/baseline.joblib"
  exit 1
fi

VIDEO_PATH="$1"
MODEL_PATH="${2:-models/baseline.joblib}"

if [[ ! -f "$VIDEO_PATH" ]]; then
  echo "ERROR: video not found: $VIDEO_PATH" >&2
  exit 2
fi

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "ERROR: model not found: $MODEL_PATH" >&2
  exit 2
fi

python -m lie_detection_expression.score_cli \
  --video "$VIDEO_PATH" \
  --model "$MODEL_PATH" \
  --out-json outputs/sentence_scores.json \
  --out-csv outputs/sentence_scores.csv

