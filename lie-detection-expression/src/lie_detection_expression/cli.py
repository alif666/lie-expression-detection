"""CLI for training baseline models from labeled OpenFace CSV files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import PipelineConfig
from .openface import resolve_openface_binary, run_openface_extraction
from .sentence_scoring import score_video_sentences, write_sentence_scores
from .train import train_and_save


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train lie-detection baseline from OpenFace CSV.")
    parser.add_argument(
        "--data",
        required=True,
        help="Path to labeled OpenFace CSV (must include `label` column).",
    )
    parser.add_argument(
        "--out",
        default="models/baseline.joblib",
        help="Path to output model file.",
    )
    parser.add_argument("--label-column", default="label", help="Label column name.")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second.")
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.7,
        help="Minimum confidence threshold for filtering OpenFace rows.",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=200,
        help="Max training epochs for MLP.",
    )
    return parser


def train_main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    config = PipelineConfig(
        fps=args.fps,
        min_confidence=args.min_confidence,
        max_epochs=args.max_epochs,
    )
    model = train_and_save(
        csv_path=Path(args.data),
        model_out_path=Path(args.out),
        config=config,
        label_column=args.label_column,
    )
    print(f"Model saved to: {args.out}")
    print(f"Feature columns: {len(model.feature_columns)}")


if __name__ == "__main__":
    train_main()


def _build_score_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score per-sentence deception probabilities from video transcript alignment."
    )
    parser.add_argument("--video", required=True, help="Path to input video (.mp4).")
    parser.add_argument(
        "--openface-csv",
        required=False,
        help="Path to OpenFace CSV generated from the same video. Defaults to <video_stem>.csv in same folder.",
    )
    parser.add_argument(
        "--openface-bin",
        required=False,
        help="Path to OpenFace FeatureExtraction executable. If omitted, OPENFACE_BIN or PATH is used.",
    )
    parser.add_argument(
        "--openface-out-dir",
        default="outputs/openface",
        help="Directory to write OpenFace outputs when CSV is missing.",
    )
    parser.add_argument(
        "--model",
        default="models/baseline.joblib",
        help="Path to trained model artifact.",
    )
    parser.add_argument(
        "--out-json",
        default="outputs/sentence_scores.json",
        help="Path to JSON output file.",
    )
    parser.add_argument(
        "--out-csv",
        default="outputs/sentence_scores.csv",
        help="Path to CSV output file.",
    )
    parser.add_argument("--whisper-model", default="base", help="Whisper model size.")
    parser.add_argument("--fps", type=int, default=30, help="Video FPS used for window size.")
    parser.add_argument("--window-seconds", type=float, default=2.0, help="Window duration.")
    parser.add_argument("--min-confidence", type=float, default=0.7, help="OpenFace confidence gate.")
    return parser


def score_main() -> None:
    parser = _build_score_parser()
    args = parser.parse_args()
    video_path = Path(args.video)
    openface_csv = Path(args.openface_csv) if args.openface_csv else video_path.with_suffix(".csv")
    if not video_path.exists():
        print(f"ERROR: video not found: {video_path}", file=sys.stderr)
        raise SystemExit(2)
    if not openface_csv.exists():
        openface_bin = resolve_openface_binary(args.openface_bin)
        if not openface_bin:
            print(
                "ERROR: OpenFace CSV not found and FeatureExtraction executable was not found.\n"
                "Provide --openface-csv or --openface-bin (or set OPENFACE_BIN).",
                file=sys.stderr,
            )
            raise SystemExit(2)
        try:
            generated = run_openface_extraction(
                video_path=video_path,
                out_dir=Path(args.openface_out_dir),
                openface_bin=openface_bin,
            )
        except RuntimeError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            raise SystemExit(2) from exc
        openface_csv = generated
        print(f"Generated OpenFace CSV: {openface_csv}")

    config = PipelineConfig(
        fps=args.fps,
        window_seconds=args.window_seconds,
        min_confidence=args.min_confidence,
    )
    try:
        payload = score_video_sentences(
            video_path=video_path,
            openface_csv_path=openface_csv,
            model_path=Path(args.model),
            config=config,
            whisper_model_name=args.whisper_model,
        )
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    write_sentence_scores(payload=payload, out_json=Path(args.out_json), out_csv=Path(args.out_csv))
    print(f"Clip probability: {payload['clip_probability']:.4f}")
    print(f"Wrote: {args.out_json}")
    print(f"Wrote: {args.out_csv}")
