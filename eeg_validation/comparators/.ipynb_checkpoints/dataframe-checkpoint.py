"""
DataFrameComparator — generic column-by-column comparison of two DataFrames.

Refactored from the old ``compare_shared_columns`` function into a reusable
class that returns a :class:`ComparisonResult`.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Union

import numpy as np
import pandas as pd

from .base import Comparator, ComparisonResult
from .utils import is_numeric_series, nan_safe_equal, nan_safe_isclose, one_unique


class DataFrameComparator(Comparator):
    """Compare all shared columns between two DataFrames.

    Parameters
    ----------
    rtol, atol : float
        Relative / absolute tolerance for numeric comparisons.
    tolerant_numeric : bool
        If True, numeric columns use ``np.isclose`` instead of exact ``==``.
    max_mismatches : int
        Max mismatch examples to store per column.
    sort_keys : list[str] | None
        Columns to sort both frames by before comparing row-by-row.
    drop_cols : set[str]
        Columns to exclude from comparison.
    allow_length_mismatch : bool
        If True, compare the overlapping prefix instead of raising.
    """

    def __init__(
        self,
        *,
        rtol: float = 1e-6,
        atol: float = 1e-8,
        tolerant_numeric: bool = True,
        max_mismatches: int = 20,
        sort_keys: Optional[Sequence[str]] = None,
        drop_cols: Iterable[str] = (),
        allow_length_mismatch: bool = False,
    ):
        self.rtol = rtol
        self.atol = atol
        self.tolerant_numeric = tolerant_numeric
        self.max_mismatches = max_mismatches
        self.sort_keys = list(sort_keys) if sort_keys else None
        self.drop_cols = set(drop_cols)
        self.allow_length_mismatch = allow_length_mismatch

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def compare(
        self,
        df_a: pd.DataFrame,
        df_b: pd.DataFrame,
        *,
        label_a: str = "A",
        label_b: str = "B",
        subject: Optional[str] = None,
        experiment: Optional[str] = None,
        session: Optional[Union[str, int]] = None,
        return_aligned: bool = False,
    ) -> ComparisonResult:
        # Infer metadata if not supplied
        subject = subject or one_unique(df_a, "subject") or one_unique(df_b, "subject")
        experiment = experiment or one_unique(df_a, "experiment") or one_unique(df_b, "experiment")
        session = session or one_unique(df_a, "session") or one_unique(df_b, "session")

        a2 = df_a.copy()
        b2 = df_b.copy()

        shared_cols = sorted(
            (set(a2.columns) & set(b2.columns)) - self.drop_cols
        )
        only_a = sorted(set(a2.columns) - set(b2.columns) - self.drop_cols)
        only_b = sorted(set(b2.columns) - set(a2.columns) - self.drop_cols)

        # Align rows
        a_aligned, b_aligned, align_mode, sort_keys_used = self._align(
            a2[shared_cols], b2[shared_cols], shared_cols
        )

        n_a, n_b = len(a_aligned), len(b_aligned)
        length_mismatch = n_a != n_b

        if length_mismatch and not self.allow_length_mismatch:
            raise AssertionError(
                f"Row count mismatch: {label_a}={n_a} vs {label_b}={n_b}"
            )

        n = min(n_a, n_b)
        a_aligned = a_aligned.iloc[:n].reset_index(drop=True)
        b_aligned = b_aligned.iloc[:n].reset_index(drop=True)

        # Column-by-column comparison
        col_rows: List[Dict[str, Any]] = []
        mismatch_rows: List[Dict[str, Any]] = []
        differing_cols: List[str] = []

        for col in shared_cols:
            sa, sb = a_aligned[col], b_aligned[col]
            used_isclose = self.tolerant_numeric and (
                is_numeric_series(sa) or is_numeric_series(sb)
            )

            ok = (
                nan_safe_isclose(sa, sb, self.rtol, self.atol)
                if used_isclose
                else nan_safe_equal(sa.astype("object"), sb.astype("object"))
            )

            n_bad = int((~ok).sum())
            if n_bad > 0:
                differing_cols.append(col)
                bad_idx = np.where(~ok)[0][: self.max_mismatches]
                for i in bad_idx:
                    mismatch_rows.append(
                        {
                            "subject": subject,
                            "experiment": experiment,
                            "session": session,
                            "column": col,
                            "i": int(i),
                            label_a: sa.iloc[i],
                            label_b: sb.iloc[i],
                        }
                    )

            col_rows.append(
                {
                    "subject": subject,
                    "experiment": experiment,
                    "session": session,
                    "column": col,
                    "n_mismatches": n_bad,
                    "fraction_mismatch": (n_bad / n) if n else np.nan,
                    "dtype_a": str(sa.dtype),
                    "dtype_b": str(sb.dtype),
                    "numeric_compared_with_isclose": used_isclose,
                }
            )

        df_detail = (
            pd.DataFrame(col_rows)
            .sort_values(["n_mismatches", "column"], ascending=[False, True])
            .reset_index(drop=True)
        )
        df_mismatches = pd.DataFrame(mismatch_rows)

        any_mismatch = bool(
            differing_cols or length_mismatch or only_a or only_b
        )

        summary = {
            "subject": subject,
            "experiment": experiment,
            "session": session,
            "comparison": f"{label_a} vs {label_b}",
            "source_a": label_a,
            "source_b": label_b,
            "n_rows_compared": int(n),
            "n_rows_a": int(n_a),
            "n_rows_b": int(n_b),
            "length_mismatch": bool(length_mismatch),
            "n_columns_compared": len(shared_cols),
            "n_differing_columns": len(differing_cols),
            "differing_columns": differing_cols,
            "n_only_in_a": len(only_a),
            "n_only_in_b": len(only_b),
            "only_in_a": only_a,
            "only_in_b": only_b,
            "any_mismatch": any_mismatch,
            "tolerant_numeric": self.tolerant_numeric,
            "numeric_rtol": self.rtol,
            "numeric_atol": self.atol,
            "sort_keys_used": sort_keys_used,
            "align_mode": align_mode,
        }

        result = ComparisonResult(
            ok=not any_mismatch,
            df_summary=pd.DataFrame([summary]),
            df_detail=df_detail,
            df_mismatches=df_mismatches,
            subject=subject,
            experiment=experiment,
            session=session,
        )

        if return_aligned:
            for df in (a_aligned, b_aligned):
                for col_name, val in [
                    ("subject", subject),
                    ("experiment", experiment),
                    ("session", session),
                ]:
                    if col_name not in df.columns:
                        df.insert(0, col_name, val)
            result.aligned_a = a_aligned
            result.aligned_b = b_aligned

        return result

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _align(self, a, b, shared_cols):
        if self.sort_keys is None:
            return (
                a.reset_index(drop=True),
                b.reset_index(drop=True),
                "index",
                [],
            )

        keys = [k for k in self.sort_keys if k in shared_cols]
        if not keys:
            return (
                a.reset_index(drop=True),
                b.reset_index(drop=True),
                "index",
                [],
            )

        return (
            a.sort_values(keys, kind="mergesort").reset_index(drop=True),
            b.sort_values(keys, kind="mergesort").reset_index(drop=True),
            "sorted",
            keys,
        )
