"""I/O utilities for loading/saving aggregated results."""

from __future__ import annotations

import os
from typing import List, Sequence

import pandas as pd


def load_and_concat(
    file_list: Sequence[str],
    *,
    remove_duplicates: bool = True,
) -> pd.DataFrame:
    """Load and concatenate CSV files, optionally deduplicating."""
    dfs = []
    for f in file_list:
        try:
            df = pd.read_csv(f)
            if not df.empty:
                dfs.append(df)
        except (pd.errors.EmptyDataError, Exception) as e:
            print(f"Warning: skipping {f}: {e}")

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    if remove_duplicates:
        before = len(combined)
        combined = combined.drop_duplicates()
        removed = before - len(combined)
        if removed:
            print(f"Removed {removed} duplicate rows")
    return combined


def delete_files(file_list: Sequence[str]) -> None:
    """Delete a list of files (best-effort)."""
    for f in file_list:
        try:
            os.remove(f)
        except Exception as e:
            print(f"Error deleting {f}: {e}")
