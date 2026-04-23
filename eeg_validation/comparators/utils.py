"""Low-level comparison helpers (NaN-safe equality, tolerance checks)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def nan_safe_equal(a: pd.Series, b: pd.Series) -> np.ndarray:
    """Element-wise equality that treats NaN == NaN as True.

    Normalises pd.NA / NaT / NaN → None before comparing so the
    elementwise `==` returns real Python booleans (pd.NA==pd.NA yields
    pd.NA, which breaks downstream bitwise ops)."""
    a_np = np.asarray(a, dtype=object)
    b_np = np.asarray(b, dtype=object)
    a_mask = pd.isna(a_np)
    b_mask = pd.isna(b_np)
    a_np = np.where(a_mask, None, a_np)
    b_np = np.where(b_mask, None, b_np)
    both_nan = a_mask & b_mask
    return (a_np == b_np) | both_nan


def nan_safe_isclose(
    a: pd.Series,
    b: pd.Series,
    rtol: float = 1e-6,
    atol: float = 1e-8,
) -> np.ndarray:
    """Element-wise isclose that treats NaN == NaN as True."""
    a_np = pd.to_numeric(a, errors="coerce").to_numpy()
    b_np = pd.to_numeric(b, errors="coerce").to_numpy()
    return np.isclose(a_np, b_np, rtol=rtol, atol=atol, equal_nan=True)


def is_numeric_series(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s)


def one_unique(df: pd.DataFrame, col: str):
    """Return the single unique non-null value in *col*, or None."""
    if col not in df.columns:
        return None
    vals = df[col].dropna().unique()
    return vals[0] if len(vals) == 1 else None


def crop_to_min_length(*arrays: np.ndarray, axis: int = -1):
    """Crop arrays along *axis* to the shortest length. Returns list + min_len."""
    min_len = min(a.shape[axis] for a in arrays)
    sliced = [np.take(a, range(min_len), axis=axis) for a in arrays]
    return (*sliced, min_len)
