"""OpenFace invocation helpers."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


def resolve_openface_binary(explicit_bin: str | None = None) -> str | None:
    """Resolve FeatureExtraction executable from arg, env, OPENFACE_DIR, or PATH."""
    if explicit_bin:
        return explicit_bin

    env_bin = os.environ.get("OPENFACE_BIN")
    if env_bin:
        return env_bin

    openface_dir = os.environ.get("OPENFACE_DIR")
    if openface_dir:
        base = Path(openface_dir)
        for rel in ("FeatureExtraction", "FeatureExtraction.exe", "build/bin/FeatureExtraction"):
            candidate = base / rel
            if candidate.exists():
                return str(candidate)

    for candidate in ("FeatureExtraction.exe", "FeatureExtraction"):
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    return None


def _locate_openface_csv(out_dir: Path, video_stem: str) -> Path:
    expected = out_dir / f"{video_stem}.csv"
    if expected.exists():
        return expected
    csvs = sorted(out_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if csvs:
        return csvs[0]
    raise RuntimeError(f"OpenFace did not produce a CSV in: {out_dir}")


def run_openface_extraction(
    video_path: str | Path,
    out_dir: str | Path,
    openface_bin: str,
) -> Path:
    """Run OpenFace FeatureExtraction and return generated CSV path."""
    video_path = Path(video_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        openface_bin,
        "-f",
        str(video_path),
        "-out_dir",
        str(out_dir),
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "OpenFace extraction failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"STDOUT:\n{proc.stdout[-2000:]}\n"
            f"STDERR:\n{proc.stderr[-2000:]}"
        )
    return _locate_openface_csv(out_dir=out_dir, video_stem=video_path.stem)
