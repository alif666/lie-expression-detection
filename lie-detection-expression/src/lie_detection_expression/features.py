"""Feature engineering for AU presence and intensity."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _au_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    presence = sorted([c for c in df.columns if c.startswith("AU") and c.endswith("_c")])
    intensity = sorted([c for c in df.columns if c.startswith("AU") and c.endswith("_r")])
    return presence, intensity


def build_frame_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build AU_presence * AU_intensity interaction features by frame."""
    presence_cols, intensity_cols = _au_columns(df)
    if not presence_cols or not intensity_cols:
        raise ValueError("No AU presence/intensity columns found in OpenFace data.")

    intensity_by_au = {col.removesuffix("_r"): col for col in intensity_cols}
    features: dict[str, np.ndarray] = {}

    for p_col in presence_cols:
        au_name = p_col.removesuffix("_c")
        i_col = intensity_by_au.get(au_name)
        if i_col is None:
            continue
        features[f"{au_name}_pxi"] = df[p_col].to_numpy(dtype=float) * df[i_col].to_numpy(dtype=float)

    if not features:
        raise ValueError("No matching AU presence/intensity pairs found.")

    return pd.DataFrame(features)

