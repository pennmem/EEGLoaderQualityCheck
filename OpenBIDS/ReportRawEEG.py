import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import cmlreaders as cml
from cmldask import CMLDask as da
from dask.distributed import wait, as_completed
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors
import seaborn as sns
import pandas as pd
import xarray as xr
import scipy as scp
import re
from scipy import stats
from ptsa.data.timeseries import *
from statsmodels.stats.multitest import multipletests
import pyedflib
from mne_bids import get_entity_vals
pd.options.display.max_rows = 100
pd.options.display.max_columns = 50
import mne
from mne_bids import BIDSPath, read_raw_bids


# ----------------------------
# USER SETTINGS
# ----------------------------
RTOL = 1e-6
ATOL = 1e-9
MAX_MISMATCHES_PER_CHANNEL = 10

PRINT_FULL_RAW_IF_SMALL = True
MAX_ELEMENTS_FOR_FULL_PRINT = 50_000
PRINT_SLICE = True
SLICE_CFG = {"event": 0, "channel": slice(0, 5), "time": slice(0, 20)}

### Compare behavioral
import numpy as np
import pandas as pd

def compare_behavioral(
    evs_cml,
    label_cml,
    evs_bids,
    label_bids,
    *,
    bids_is_eeg=True,
    options=None,
    rtol=RTOL,
    atol=ATOL,
    max_mismatches=20,
    drop_cols=(),
):
    """
    Compare CML vs BIDS behavioral/event tables allowing for known column name/definition differences:
      - eegoffset -> sample
      - mstime -> onset  (CML ms -> seconds)
      - type -> trial_type

    Then compares ALL other shared columns too.

    New option:
      - "compare_onset_as_diff" : compare onset as diff() in both sources (inter-event intervals)
                                 else compare absolute onset (CML shifted to start at 0; BIDS as-is)

    Other Options:
      - "align_by_index"              : don't sort; compare row order as-is
      - "allow_length_mismatch"       : compare only min(n_cml, n_bids)
      - "tolerant_numeric"            : numeric cols compared with isclose(rtol/atol) (default ON)
      - "print_behavior_summary"      : print 1-row summary
      - "print_behavior_col_summary"  : print per-column mismatch counts
      - "print_behavior_mismatches"   : print example mismatching cells (long format)
      - "return_aligned"              : return aligned comparison tables
      - "return_col_summary"          : return per-column summary df
      - "return_mismatches"           : return mismatch df (example cells)
    """
    options_set = set(options or ())
    tolerant_numeric = ("tolerant_numeric" in options_set) or ("tolerant_onset" in options_set)  # backward compat
    drop_cols = set(drop_cols or ())

    # ---------- helpers ----------
    def _one_unique(df, col, label):
        if col not in df.columns:
            return None
        vals = df[col].dropna().unique()
        if len(vals) == 0:
            return None
        if len(vals) != 1:
            raise ValueError(f"{label}: column '{col}' has {len(vals)} unique values: {vals[:10]}")
        return vals[0]

    def _is_numeric_series(s: pd.Series) -> bool:
        return pd.api.types.is_numeric_dtype(s)

    def _nan_safe_equal(a: pd.Series, b: pd.Series) -> np.ndarray:
        a = a.to_numpy()
        b = b.to_numpy()
        both_nan = pd.isna(a) & pd.isna(b)
        return (a == b) | both_nan

    def _nan_safe_isclose(a: pd.Series, b: pd.Series, *, rtol_use, atol_use) -> np.ndarray:
        a = pd.to_numeric(a, errors="coerce").to_numpy()
        b = pd.to_numeric(b, errors="coerce").to_numpy()
        return np.isclose(a, b, rtol=rtol_use, atol=atol_use, equal_nan=True)

    # ---------- subject/experiment/session checks ----------
    subject_cml = _one_unique(evs_cml, "subject", label_cml)
    subject_bids = _one_unique(evs_bids, "subject", label_bids)
    experiment_cml = _one_unique(evs_cml, "experiment", label_cml)
    experiment_bids = _one_unique(evs_bids, "experiment", label_bids)
    session_cml = _one_unique(evs_cml, "session", label_cml)
    session_bids = _one_unique(evs_bids, "session", label_bids)

    if subject_cml is not None and subject_bids is not None and subject_cml != subject_bids:
        raise ValueError(f"Subjects differ: {label_cml}={subject_cml} vs {label_bids}={subject_bids}")
    if experiment_cml is not None and experiment_bids is not None and experiment_cml != experiment_bids:
        raise ValueError(f"Experiments differ: {label_cml}={experiment_cml} vs {label_bids}={experiment_bids}")
    if session_cml is not None and session_bids is not None and session_cml != session_bids:
        raise ValueError(f"Sessions differ: {label_cml}={session_cml} vs {label_bids}={session_bids}")

    subject = subject_cml if subject_cml is not None else subject_bids
    experiment = experiment_cml if experiment_cml is not None else experiment_bids
    session = session_cml if session_cml is not None else session_bids

    # ---------- required cols ----------
    if bids_is_eeg:
        required_cml = {"eegoffset", "mstime", "type"}
        required_bids = {"sample", "onset", "trial_type"}

        missing_cml = required_cml - set(evs_cml.columns)
        missing_bids = required_bids - set(evs_bids.columns)
        if missing_cml:
            raise ValueError(f"{label_cml}: missing required columns: {sorted(missing_cml)}")
        if missing_bids:
            raise ValueError(f"{label_bids}: missing required columns: {sorted(missing_bids)}")

        # ---------- normalize CML to BIDS-like names ----------
        cml2 = evs_cml.copy()
        bids2 = evs_bids.copy()

        # CML sentinel missing -> NaN (and common empty-string missing)
        cml2 = cml2.replace({-999: np.nan, -999.0: np.nan, "-999": np.nan, "": np.nan})

        # rename to match BIDS schema for the 3 key cols
        cml2 = cml2.rename(columns={"eegoffset": "sample", "mstime": "onset", "type": "trial_type"})

        # ensure mapped cols numeric/string comparable
        cml2["sample"] = pd.to_numeric(cml2["sample"], errors="raise")
        bids2["sample"] = pd.to_numeric(bids2["sample"], errors="raise")
        cml2["trial_type"] = cml2["trial_type"].astype(str)
        bids2["trial_type"] = bids2["trial_type"].astype(str)

        # ---- onset construction (NEW) ----
        # Convert CML onset (ms) -> seconds; BIDS onset assumed seconds
        cml_onset_s = pd.to_numeric(evs_cml["mstime"], errors="raise") / 1000.0
        bids_onset_s = pd.to_numeric(bids2["onset"], errors="raise")

        if "compare_onset_as_diff" in options_set:
            # Compare inter-event intervals
            cml2["onset"] = cml_onset_s.diff()
            print(cml2["onset"])
            bids2["onset"] = bids_onset_s.diff()
            print(bids2["onset"])
        else:
            # Compare absolute onset (CML shifted to start at 0)
            cml2["onset"] = cml_onset_s - cml_onset_s.iloc[0]
            print(cml2["onset"])
            bids2["onset"] = bids_onset_s
            print(bids2["onset"])

    # ---------- choose columns to compare (ALL shared columns) ----------
    shared_cols = sorted((set(cml2.columns) & set(bids2.columns)) - drop_cols)

    only_cml = sorted(set(cml2.columns) - set(bids2.columns) - drop_cols)
    only_bids = sorted(set(bids2.columns) - set(cml2.columns) - drop_cols)

    # ---------- align rows ----------
    if "align_by_index" in options_set:
        cml_aligned = cml2[shared_cols].reset_index(drop=True)
        bids_aligned = bids2[shared_cols].reset_index(drop=True)
    else:
        # robust tie-break for duplicate samples
        tie_keys = [k for k in ["sample", "trial_type"] if k in shared_cols]
        # only use onset as a tie-break when NOT comparing diff(onset)
        if "compare_onset_as_diff" not in options_set and "onset" in shared_cols:
            tie_keys.append("onset")

        cml_aligned = cml2[shared_cols].sort_values(tie_keys, kind="mergesort").reset_index(drop=True)
        bids_aligned = bids2[shared_cols].sort_values(tie_keys, kind="mergesort").reset_index(drop=True)

    n_cml = len(cml_aligned)
    n_bids = len(bids_aligned)
    length_mismatch = (n_cml != n_bids)

    if length_mismatch and ("allow_length_mismatch" not in options_set):
        raise AssertionError(f"Event count mismatch: {label_cml}={n_cml} vs {label_bids}={n_bids}")

    n = min(n_cml, n_bids)
    cml_aligned = cml_aligned.iloc[:n].reset_index(drop=True)
    bids_aligned = bids_aligned.iloc[:n].reset_index(drop=True)

    # ---------- per-column comparison ----------
    col_rows = []
    differing_cols = []
    mismatch_examples = []
    print(shared_cols)
    for col in shared_cols:
        a = cml_aligned[col]
        b = bids_aligned[col]

        if tolerant_numeric and (_is_numeric_series(a) or _is_numeric_series(b)):
            rtol_use, atol_use = rtol, atol

            # Special-case onset in seconds: allow ms-level tolerance
            if col == "onset":
                rtol_use, atol_use = 0.0, 0.002  # 2 ms

            ok = _nan_safe_isclose(a, b, rtol_use=rtol_use, atol_use=atol_use)
        else:
            ok = _nan_safe_equal(a.astype("object"), b.astype("object"))

        n_bad = int((~ok).sum())
        if n_bad > 0:
            differing_cols.append(col)
            bad_idx = np.where(~ok)[0][:max_mismatches]
            for i in bad_idx:
                mismatch_examples.append({
                    "column": col,
                    "i": int(i),
                    f"{label_cml}": a.iloc[i],
                    f"{label_bids}": b.iloc[i],
                })

        col_rows.append({
            "column": col,
            "n_mismatches": n_bad,
            "fraction_mismatch": (n_bad / n) if n else np.nan,
            "dtype_cml": str(a.dtype),
            "dtype_bids": str(b.dtype),
        })

    df_col_summary = (
        pd.DataFrame(col_rows)
        .sort_values(["n_mismatches", "column"], ascending=[False, True])
        .reset_index(drop=True)
    )
    df_mismatches = pd.DataFrame(mismatch_examples)

    # ---------- summary ----------
    summary = dict(
        subject=subject,
        experiment=experiment,
        session=session,

        comparison=f"{label_cml} vs {label_bids}",
        source_a=label_cml,
        source_b=label_bids,

        n_events_compared=int(n),
        n_events_cml=int(n_cml),
        n_events_bids=int(n_bids),
        length_mismatch=bool(length_mismatch),

        n_columns_compared=int(len(shared_cols)),
        n_differing_columns=int(len(differing_cols)),
        differing_columns=differing_cols,

        n_only_in_cml=int(len(only_cml)),
        n_only_in_bids=int(len(only_bids)),
        only_in_cml=only_cml,
        only_in_bids=only_bids,

        any_mismatch=bool(
            (len(differing_cols) > 0) or length_mismatch or (len(only_cml) > 0) or (len(only_bids) > 0)
        ),
        numeric_rtol=float(rtol) if tolerant_numeric else 0.0,
        numeric_atol=float(atol) if tolerant_numeric else 0.0,

        onset_mode="diff" if "compare_onset_as_diff" in options_set else "absolute",
    )
    df_summary = pd.DataFrame([summary])

    # ---------- optional prints ----------
    if "print_behavior_summary" in options_set:
        print("\n================ BEHAVIOR SUMMARY ================")
        print(df_summary.to_string(index=False))

    if "print_behavior_col_summary" in options_set:
        print("\n================ BEHAVIOR PER-COLUMN MISMATCH COUNTS ================")
        print(df_col_summary.to_string(index=False))

    if "print_behavior_mismatches" in options_set:
        print("\n================ BEHAVIOR MISMATCH EXAMPLES (first few) ================")
        if len(df_mismatches) == 0:
            print("[OK] No mismatches.")
        else:
            print(df_mismatches.head(max_mismatches).to_string(index=False))

    # ---------- return payload ----------
    out = {"df_behavior_summary": df_summary, "ok": not summary["any_mismatch"]}

    if "return_col_summary" in options_set:
        out["df_behavior_column_summary"] = df_col_summary

    if "return_mismatches" in options_set:
        out["df_behavior_mismatches"] = df_mismatches

    if "return_aligned" in options_set:
        out["cml_aligned"] = cml_aligned
        out["bids_aligned"] = bids_aligned

    return out


import numpy as np
import pandas as pd
from typing import Iterable, Optional, Sequence, Union, Dict, Any

def compare_shared_columns(
    df_a: pd.DataFrame,
    label_a: str,
    df_b: pd.DataFrame,
    label_b: str,
    *,
    options: Optional[Iterable[str]] = None,
    tolerant_numeric: Optional[bool] = None,
    rtol: float = 1e-6,
    atol: float = 1e-8,
    max_mismatches: int = 20,
    drop_cols: Union[Sequence[str], set, tuple] = (),
    sort_keys: Optional[Sequence[str]] = None,
    allow_length_mismatch: bool = False,
    summary_outfile: Optional[str] = None,

    # NEW: allow caller to force metadata (recommended)
    subject: Optional[str] = None,
    experiment: Optional[str] = None,
    session: Optional[Union[str, int]] = None,
) -> Dict[str, Any]:
    """
    Generic comparator: compares all shared columns between df_a and df_b.

    NEW:
      subject/experiment/session are included in ALL returned tables:
        - df_summary
        - df_column_summary
        - df_mismatches
        - a_aligned/b_aligned (if returned)

    If not provided, tries to infer single-valued subject/experiment/session
    from either df.
    """
    options_set = set(options or ())
    drop_cols = set(drop_cols or ())

    if tolerant_numeric is None:
        tolerant_numeric = ("tolerant_numeric" in options_set)

    allow_len = bool(allow_length_mismatch or ("allow_length_mismatch" in options_set))

    def _one_unique(df: pd.DataFrame, col: str) -> Optional[Any]:
        if col not in df.columns:
            return None
        vals = df[col].dropna().unique()
        if len(vals) == 0:
            return None
        if len(vals) != 1:
            return None
        return vals[0]

    # Infer metadata if caller didn't provide it
    if subject is None:
        subject = _one_unique(df_a, "subject") or _one_unique(df_b, "subject")
    if experiment is None:
        experiment = _one_unique(df_a, "experiment") or _one_unique(df_b, "experiment")
    if session is None:
        session = _one_unique(df_a, "session") or _one_unique(df_b, "session")

    def _is_numeric_series(s: pd.Series) -> bool:
        return pd.api.types.is_numeric_dtype(s)

    def _nan_safe_equal(a: pd.Series, b: pd.Series) -> np.ndarray:
        a = a.to_numpy()
        b = b.to_numpy()
        both_nan = pd.isna(a) & pd.isna(b)
        return (a == b) | both_nan

    def _nan_safe_isclose(a: pd.Series, b: pd.Series) -> np.ndarray:
        a = pd.to_numeric(a, errors="coerce").to_numpy()
        b = pd.to_numeric(b, errors="coerce").to_numpy()
        return np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=True)

    a2 = df_a.copy()
    b2 = df_b.copy()

    shared_cols = sorted((set(a2.columns) & set(b2.columns)) - drop_cols)
    only_a = sorted(set(a2.columns) - set(b2.columns) - drop_cols)
    only_b = sorted(set(b2.columns) - set(a2.columns) - drop_cols)

    # align rows
    if "align_by_index" in options_set or sort_keys is None:
        a_aligned = a2[shared_cols].reset_index(drop=True)
        b_aligned = b2[shared_cols].reset_index(drop=True)
        align_mode = "index"
        sort_keys_used = []
    else:
        keys = [k for k in sort_keys if k in shared_cols]
        if len(keys) == 0:
            a_aligned = a2[shared_cols].reset_index(drop=True)
            b_aligned = b2[shared_cols].reset_index(drop=True)
            align_mode = "index"
            sort_keys_used = []
        else:
            a_aligned = a2[shared_cols].sort_values(keys, kind="mergesort").reset_index(drop=True)
            b_aligned = b2[shared_cols].sort_values(keys, kind="mergesort").reset_index(drop=True)
            align_mode = "sorted"
            sort_keys_used = keys

    n_a = len(a_aligned)
    n_b = len(b_aligned)
    length_mismatch = (n_a != n_b)

    if length_mismatch and not allow_len:
        raise AssertionError(f"Row count mismatch: {label_a}={n_a} vs {label_b}={n_b}")

    n = min(n_a, n_b)
    a_aligned = a_aligned.iloc[:n].reset_index(drop=True)
    b_aligned = b_aligned.iloc[:n].reset_index(drop=True)

    col_rows = []
    differing_cols = []
    mismatch_examples = []

    for col in shared_cols:
        sa = a_aligned[col]
        sb = b_aligned[col]

        used_isclose = bool(tolerant_numeric and (_is_numeric_series(sa) or _is_numeric_series(sb)))
        if used_isclose:
            ok = _nan_safe_isclose(sa, sb)
        else:
            ok = _nan_safe_equal(sa.astype("object"), sb.astype("object"))

        n_bad = int((~ok).sum())
        if n_bad > 0:
            differing_cols.append(col)
            bad_idx = np.where(~ok)[0][:max_mismatches]
            for i in bad_idx:
                mismatch_examples.append({
                    "subject": subject,
                    "experiment": experiment,
                    "session": session,
                    "column": col,
                    "i": int(i),
                    label_a: sa.iloc[i],
                    label_b: sb.iloc[i],
                })

        col_rows.append({
            "subject": subject,
            "experiment": experiment,
            "session": session,
            "column": col,
            "n_mismatches": n_bad,
            "fraction_mismatch": (n_bad / n) if n else np.nan,
            "dtype_a": str(sa.dtype),
            "dtype_b": str(sb.dtype),
            "numeric_compared_with_isclose": used_isclose,
        })

    df_column_summary = (
        pd.DataFrame(col_rows)
        .sort_values(["n_mismatches", "column"], ascending=[False, True])
        .reset_index(drop=True)
    )
    df_mismatches = pd.DataFrame(mismatch_examples)

    summary = dict(
        subject=subject,
        experiment=experiment,
        session=session,

        comparison=f"{label_a} vs {label_b}",
        source_a=label_a,
        source_b=label_b,

        n_rows_compared=int(n),
        n_rows_a=int(n_a),
        n_rows_b=int(n_b),
        length_mismatch=bool(length_mismatch),

        n_columns_compared=int(len(shared_cols)),
        n_differing_columns=int(len(differing_cols)),
        differing_columns=differing_cols,

        n_only_in_a=int(len(only_a)),
        n_only_in_b=int(len(only_b)),
        only_in_a=only_a,
        only_in_b=only_b,

        any_mismatch=bool(
            (len(differing_cols) > 0) or length_mismatch or (len(only_a) > 0) or (len(only_b) > 0)
        ),

        tolerant_numeric=bool(tolerant_numeric),
        numeric_rtol=float(rtol) if tolerant_numeric else 0.0,
        numeric_atol=float(atol) if tolerant_numeric else 0.0,
        sort_keys_used=sort_keys_used,
        align_mode=align_mode,
    )
    df_summary = pd.DataFrame([summary])

    # optional prints
    if "print_summary" in options_set:
        print("\n================ SUMMARY ================")
        print(df_summary.to_string(index=False))

    if "print_col_summary" in options_set:
        print("\n================ PER-COLUMN MISMATCH COUNTS ================")
        print(df_column_summary.to_string(index=False))

    if "print_mismatches" in options_set:
        print("\n================ MISMATCH EXAMPLES ================")
        if len(df_mismatches) == 0:
            print("[OK] No mismatches.")
        else:
            print(df_mismatches.head(max_mismatches).to_string(index=False))

    if summary_outfile is not None:
        df_summary.to_csv(summary_outfile, index=False)

    out: Dict[str, Any] = {
        "df_summary": df_summary,
        "df_column_summary": df_column_summary,
        "df_mismatches": df_mismatches,
        "ok": not summary["any_mismatch"],
    }

    if "return_aligned" in options_set:
        a_out = a_aligned.copy()
        b_out = b_aligned.copy()
        a_out.insert(0, "session", session)
        a_out.insert(0, "experiment", experiment)
        a_out.insert(0, "subject", subject)
        b_out.insert(0, "session", session)
        b_out.insert(0, "experiment", experiment)
        b_out.insert(0, "subject", subject)
        out["a_aligned"] = a_out
        out["b_aligned"] = b_out

    return out


# def compare_shared_columns(
#     df_a: pd.DataFrame,
#     label_a: str,
#     df_b: pd.DataFrame,
#     label_b: str,
#     *,
#     options: Optional[Iterable[str]] = None,
#     tolerant_numeric: Optional[bool] = None,
#     rtol: float = 1e-6,
#     atol: float = 1e-8,
#     max_mismatches: int = 20,
#     drop_cols: Union[Sequence[str], set, tuple] = (),
#     sort_keys: Optional[Sequence[str]] = None,
#     allow_length_mismatch: bool = False,
#     summary_outfile: Optional[str] = None,
# ) -> Dict[str, Any]:
#     """
#     Generic comparator: compares all shared columns between df_a and df_b.

#     Features:
#       - Numeric columns: compare with np.isclose (rtol/atol) when tolerant_numeric=True
#       - Non-numeric: compare with exact equality, NaN-safe
#       - Row alignment:
#           * options contains "align_by_index": compare row order as-is
#           * else: if sort_keys provided, sort by those columns (stable) before comparing
#             (sort_keys are filtered to shared columns)
#       - Length mismatch handling:
#           * if allow_length_mismatch=True (or option "allow_length_mismatch"), compare min(n_a, n_b)
#           * else raise AssertionError if lengths differ

#     Outputs:
#       - df_summary: 1-row summary dataframe
#       - df_column_summary: per-column mismatch counts
#       - df_mismatches: long-format mismatch examples
#       - optionally aligned dfs when "return_aligned" in options
#     """
#     options_set = set(options or ())
#     drop_cols = set(drop_cols or ())

#     if tolerant_numeric is None:
#         tolerant_numeric = ("tolerant_numeric" in options_set)

#     if allow_length_mismatch or ("allow_length_mismatch" in options_set):
#         allow_len = True
#     else:
#         allow_len = False

#     def _is_numeric_series(s: pd.Series) -> bool:
#         return pd.api.types.is_numeric_dtype(s)

#     def _nan_safe_equal(a: pd.Series, b: pd.Series) -> np.ndarray:
#         a = a.to_numpy()
#         b = b.to_numpy()
#         both_nan = pd.isna(a) & pd.isna(b)
#         return (a == b) | both_nan

#     def _nan_safe_isclose(a: pd.Series, b: pd.Series) -> np.ndarray:
#         a = pd.to_numeric(a, errors="coerce").to_numpy()
#         b = pd.to_numeric(b, errors="coerce").to_numpy()
#         return np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=True)

#     a2 = df_a.copy()
#     b2 = df_b.copy()

#     # choose columns to compare
#     shared_cols = sorted((set(a2.columns) & set(b2.columns)) - drop_cols)
#     only_a = sorted(set(a2.columns) - set(b2.columns) - drop_cols)
#     only_b = sorted(set(b2.columns) - set(a2.columns) - drop_cols)

#     # align rows
#     if "align_by_index" in options_set or sort_keys is None:
#         a_aligned = a2[shared_cols].reset_index(drop=True)
#         b_aligned = b2[shared_cols].reset_index(drop=True)
#     else:
#         # use only keys that actually exist in shared cols
#         keys = [k for k in sort_keys if k in shared_cols]
#         if len(keys) == 0:
#             # fall back to index alignment if no usable keys
#             a_aligned = a2[shared_cols].reset_index(drop=True)
#             b_aligned = b2[shared_cols].reset_index(drop=True)
#         else:
#             a_aligned = a2[shared_cols].sort_values(keys, kind="mergesort").reset_index(drop=True)
#             b_aligned = b2[shared_cols].sort_values(keys, kind="mergesort").reset_index(drop=True)

#     n_a = len(a_aligned)
#     n_b = len(b_aligned)
#     length_mismatch = (n_a != n_b)

#     if length_mismatch and not allow_len:
#         raise AssertionError(f"Row count mismatch: {label_a}={n_a} vs {label_b}={n_b}")

#     n = min(n_a, n_b)
#     a_aligned = a_aligned.iloc[:n].reset_index(drop=True)
#     b_aligned = b_aligned.iloc[:n].reset_index(drop=True)

#     # compare per column
#     col_rows = []
#     differing_cols = []
#     mismatch_examples = []

#     for col in shared_cols:
#         sa = a_aligned[col]
#         sb = b_aligned[col]

#         if tolerant_numeric and (_is_numeric_series(sa) or _is_numeric_series(sb)):
#             ok = _nan_safe_isclose(sa, sb)
#         else:
#             ok = _nan_safe_equal(sa.astype("object"), sb.astype("object"))

#         n_bad = int((~ok).sum())
#         if n_bad > 0:
#             differing_cols.append(col)
#             bad_idx = np.where(~ok)[0][:max_mismatches]
#             for i in bad_idx:
#                 mismatch_examples.append({
#                     "column": col,
#                     "i": int(i),
#                     label_a: sa.iloc[i],
#                     label_b: sb.iloc[i],
#                 })

#         col_rows.append({
#             "column": col,
#             "n_mismatches": n_bad,
#             "fraction_mismatch": (n_bad / n) if n else np.nan,
#             "dtype_a": str(sa.dtype),
#             "dtype_b": str(sb.dtype),
#             "numeric_compared_with_isclose": bool(
#                 tolerant_numeric and (_is_numeric_series(sa) or _is_numeric_series(sb))
#             ),
#         })

#     df_column_summary = (
#         pd.DataFrame(col_rows)
#         .sort_values(["n_mismatches", "column"], ascending=[False, True])
#         .reset_index(drop=True)
#     )
#     df_mismatches = pd.DataFrame(mismatch_examples)

#     summary = dict(
#         comparison=f"{label_a} vs {label_b}",
#         source_a=label_a,
#         source_b=label_b,

#         n_rows_compared=int(n),
#         n_rows_a=int(n_a),
#         n_rows_b=int(n_b),
#         length_mismatch=bool(length_mismatch),

#         n_columns_compared=int(len(shared_cols)),
#         n_differing_columns=int(len(differing_cols)),
#         differing_columns=differing_cols,

#         n_only_in_a=int(len(only_a)),
#         n_only_in_b=int(len(only_b)),
#         only_in_a=only_a,
#         only_in_b=only_b,

#         any_mismatch=bool(
#             (len(differing_cols) > 0) or length_mismatch or (len(only_a) > 0) or (len(only_b) > 0)
#         ),

#         tolerant_numeric=bool(tolerant_numeric),
#         numeric_rtol=float(rtol) if tolerant_numeric else 0.0,
#         numeric_atol=float(atol) if tolerant_numeric else 0.0,
#         sort_keys_used=[k for k in (sort_keys or []) if k in shared_cols] if ("align_by_index" not in options_set) else [],
#         align_mode="index" if ("align_by_index" in options_set or sort_keys is None) else "sorted",
#     )
#     df_summary = pd.DataFrame([summary])

#     # optional prints
#     if "print_summary" in options_set:
#         print("\n================ SUMMARY ================")
#         print(df_summary.to_string(index=False))

#     if "print_col_summary" in options_set:
#         print("\n================ PER-COLUMN MISMATCH COUNTS ================")
#         print(df_column_summary.to_string(index=False))

#     if "print_mismatches" in options_set:
#         print("\n================ MISMATCH EXAMPLES ================")
#         if len(df_mismatches) == 0:
#             print("[OK] No mismatches.")
#         else:
#             print(df_mismatches.head(max_mismatches).to_string(index=False))

#     # optional save summary
#     if summary_outfile is not None:
#         df_summary.to_csv(summary_outfile, index=False)

#     out: Dict[str, Any] = {
#         "df_summary": df_summary,
#         "df_column_summary": df_column_summary,
#         "df_mismatches": df_mismatches,
#         "ok": not summary["any_mismatch"],
#     }

#     if "return_aligned" in options_set:
#         out["a_aligned"] = a_aligned
#         out["b_aligned"] = b_aligned

#     return out


# ----------------------------
# HELPERS
# ----------------------------
def load_bdf_as_xarray(path: str, *, event_dim_name="event") -> xr.DataArray:
    f = pyedflib.EdfReader(path)
    try:
        n_channels = f.signals_in_file
        ch_names = list(f.getSignalLabels())
        sfreq = float(f.getSampleFrequency(0))
        n_samples = int(f.getNSamples()[0])

        # read + convert each channel explicitly to volts
        data_V = []
        for ch in range(n_channels):
            # print(ch)
            x = f.readSignal(ch)
            unit = f.getPhysicalDimension(ch).lower()

            if unit in ("uv", "µv"):
                x = x * 1e-6          # µV → V
            elif unit == "mv":
                x = x * 1e-3          # mV → V
            elif unit == "v":
                pass                 # already volts
            else:
                print(f"Unknown or invalid physical unit '{unit}' \n" + f"for channel {ch} ({ch_names[ch]})")

            data_V.append(x)

        # shape: (channel, time)
        data = np.vstack(data_V)

        times = np.arange(n_samples) / sfreq

        # add singleton event dimension
        data = data[None, :, :]  # (event=1, channel, time)

        da = xr.DataArray(
            data,
            dims=(event_dim_name, "channel", "time"),
            coords={
                event_dim_name: [0],
                "channel": ch_names,
                "time": times,
                "samplerate": sfreq,
            },
            name="eeg",
            attrs={
                "units": "V",
                "source": "pyedflib (explicitly converted to volts)",
            },
        )

        return da

    finally:
        f.close()
        
    
def strip_event_metadata(da: xr.DataArray) -> xr.DataArray:
    if "event" not in da.dims:
        return da
    drop = [c for c in da.coords if ("event" in da.coords[c].dims and c != "event")]
    return da.drop_vars(drop) if drop else da


def summarize_channel_coords(a: xr.DataArray, b: xr.DataArray, channel_dim: str):
    a_ch = list(a[channel_dim].values) if channel_dim in a.dims else []
    b_ch = list(b[channel_dim].values) if channel_dim in b.dims else []

    set_a, set_b = set(a_ch), set(b_ch)
    only_a = sorted(set_a - set_b)
    only_b = sorted(set_b - set_a)
    common = [ch for ch in a_ch if ch in set_b]  # preserve a's order for common

    print("\n--- Channel coord check ---")
    print("BIDS #channels:", len(a_ch))
    print("CML  #channels:", len(b_ch))
    print("Common channels:", len(common))
    if only_a:
        print("Only in BIDS (first 20):", only_a[:20])
    if only_b:
        print("Only in CML  (first 20):", only_b[:20])

    # order check on common channels
    b_common_in_b_order = [ch for ch in b_ch if ch in set_a]
    if common != b_common_in_b_order:
        print("Channel order differs across objects (on common channels).")
    else:
        print("Channel order matches (on common channels).")

    return a_ch, b_ch, common


def report_mismatched_channels(channel_name, da_a: xr.DataArray, da_b: xr.DataArray, rtol: float, atol: float, max_n: int):
    """
    Prints first max_n mismatch locations for:
      - exact mismatch (NaN-safe)
      - allclose mismatch (rtol/atol, NaN-safe)

    Works for 1D or ND arrays.
    """
    a = np.asarray(da_a.data)
    b = np.asarray(da_b.data)

    # squeeze singleton dims so (1, T) -> (T,)
    a = np.squeeze(a)
    b = np.squeeze(b)

    if a.shape != b.shape:
        print(f"\n[{channel_name}] cannot report mismatches: shape bids {a.shape} vs cml {b.shape}")
        return

    both_nan = np.isnan(a) & np.isnan(b)
    exact_bad = ~((a == b) | both_nan)
    close_bad = ~np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=True)

    def _print_bad(mask, label):
        if not np.any(mask):
            return
        print(f"\n[{channel_name}] {label} mismatches (up to {max_n})")
        inds = np.argwhere(mask)

        for idx_arr in inds[:max_n]:
            idx = tuple(idx_arr.tolist())

            # coords: try to print time (common case)
            coord_info = {}
            if da_a.ndim == 1 and "time" in da_a.dims:
                # idx is (time_index,)
                ti = idx[0]
                try:
                    coord_info["time"] = float(da_a["time"].values[ti])
                except Exception:
                    coord_info["time_index"] = ti
            else:
                # ND case: map each dim
                for d, i in zip(da_a.dims, idx):
                    if d in da_a.coords and da_a.coords[d].ndim == 1:
                        try:
                            coord_info[d] = da_a.coords[d].values[i]
                        except Exception:
                            coord_info[f"{d}_index"] = i
                    else:
                        coord_info[f"{d}_index"] = i

            aval = a[idx]
            bval = b[idx]
            if label == "ALLCLOSE":
                err = np.abs(aval - bval)
                print(" index:", idx, "coords:", coord_info, "bids:", aval, "cml:", bval, "abs_err:", err)
            else:
                print(" index:", idx, "coords:", coord_info, "bids:", aval, "cml:", bval)

    # _print_bad(exact_bad, "EXACT")
    # _print_bad(close_bad, "ALLCLOSE")


def print_raw_data(name: str, da: xr.DataArray):
    data = da.data
    n_elem = data.size

    print(f"\n--- {name} raw data ---")
    print("dims:", da.dims)
    print("shape:", da.shape)

    if PRINT_FULL_RAW_IF_SMALL and n_elem <= MAX_ELEMENTS_FOR_FULL_PRINT:
        print(data)
        return

    if PRINT_SLICE:
        sel = {dim: spec for dim, spec in SLICE_CFG.items() if dim in da.dims}
        try:
            sliced = da.isel(**sel)
            print(f"(showing slice {sel})")
            print(sliced.data)
        except Exception as e:
            print(f"(could not slice with {sel}: {e})")
            flat = data.ravel()
            print("(printing first 200 flattened values)")
            print(flat[: min(200, flat.size)])
    else:
        print("(raw data printing disabled; set PRINT_SLICE or PRINT_FULL_RAW_IF_SMALL)")

def _ensure_dims(da: xr.DataArray, *, event_dim="event", channel_dim="channel", time_dim="time") -> xr.DataArray:
    if event_dim not in da.dims:
        da = da.expand_dims({event_dim: [0]})
    for d in (event_dim, channel_dim, time_dim):
        if d not in da.dims:
            raise ValueError(f"Expected dim '{d}' not found. Have dims={da.dims}")
    return da.transpose(event_dim, channel_dim, time_dim)

def _crop_time_to_min(a: np.ndarray, b: np.ndarray):
    """
    Crop two arrays along the last axis to the min length.
    Returns cropped arrays and min_len.
    """
    la = a.shape[-1]
    lb = b.shape[-1]
    m = min(la, lb)
    return a[..., :m], b[..., :m], m

def compare_raw_signal_pairs(label_a: str, da_a: xr.DataArray, label_b: str, da_b: xr.DataArray):
    """
    Now supports:
      - raw:   (event=1, channel, time) or (channel, time) after _ensure_dims()
      - epoched: (event>1, channel, time)

    Returns:
      df (rows per channel x event),
      exact_fail_channels (kept for backward compat; channel has ANY exact mismatch in ANY event),
      close_fail_channels (channel has ANY close mismatch in ANY event),
      common_ch,
      close_diff_event_indices (events where ANY channel mismatches close),
      n_events
    """
    rows = []
    exact_fail = []
    close_fail = []

    # Only compare channels that exist in BOTH
    common_ch = np.intersect1d(
        da_a["channel"].astype(str).values,
        da_b["channel"].astype(str).values,
    )

    # event count (assumes both have event dim; your pipeline uses _ensure_dims upstream)
    n_events = int(min(da_a.sizes.get("event", 1), da_b.sizes.get("event", 1)))

    # Track event-level close mismatch across ALL channels
    any_close_mismatch_by_event = np.zeros(n_events, dtype=bool)

    for ch in common_ch:
        da1 = da_a.sel({"channel": ch})
        da2 = da_b.sel({"channel": ch})

        a = np.asarray(da1.data)
        b = np.asarray(da2.data)

        # squeeze channel dim away; expect (event, time) or (time,)
        a = np.squeeze(a)
        b = np.squeeze(b)

        # ensure 2D: (event, time)
        if a.ndim == 1:
            a = a[None, :]
        if b.ndim == 1:
            b = b[None, :]

        # crop to min events and min time
        E = min(a.shape[0], b.shape[0], n_events)
        a = a[:E]
        b = b[:E]

        a2, b2, m = _crop_time_to_min(a, b)  # crops last axis (time)
        # shapes now (E, m)

        both_nan = np.isnan(a2) & np.isnan(b2)
        exact_bad = ~((a2 == b2) | both_nan)
        close_bad = ~np.isclose(a2, b2, rtol=RTOL, atol=ATOL, equal_nan=True)

        # event-level aggregation for THIS channel
        any_exact_this_ch = bool(np.any(exact_bad))
        any_close_this_ch = bool(np.any(close_bad))

        if any_exact_this_ch:
            exact_fail.append(ch)
        if any_close_this_ch:
            close_fail.append(ch)

        # update global event flags: event differs if ANY timepoint differs for this channel
        any_close_mismatch_by_event[:E] |= np.any(close_bad, axis=1)

        # per-event stats for this channel
        diff = a2 - b2
        invalid = both_nan | np.isnan(a2) | np.isnan(b2)
        diff = np.where(invalid, np.nan, diff)
        abs_diff = np.abs(diff)

        n_close_per_event = close_bad.sum(axis=1).astype(int)

        mean_abs = np.nanmean(abs_diff, axis=1)
        max_abs  = np.nanmax(abs_diff, axis=1)
        mean_signed = np.nanmean(diff, axis=1)
        std_diff = np.nanstd(diff, axis=1)
        mse_channel = np.nanmean(diff**2, axis=1)

        for ev_i in range(E):
            rows.append(dict(
                comparison=f"{label_a} vs {label_b}",
                channel=str(ch),
                event=int(ev_i),
                n_close_diff=int(n_close_per_event[ev_i]),
                mean_abs_diff=float(mean_abs[ev_i]) if np.isfinite(mean_abs[ev_i]) else np.nan,
                max_abs_diff=float(max_abs[ev_i]) if np.isfinite(max_abs[ev_i]) else np.nan,
                mean_signed_diff=float(mean_signed[ev_i]) if np.isfinite(mean_signed[ev_i]) else np.nan,
                std_diff=float(std_diff[ev_i]) if np.isfinite(std_diff[ev_i]) else np.nan,
                mse_channel=float(mse_channel[ev_i]) if np.isfinite(mse_channel[ev_i]) else np.nan,
                time_compared_samples=int(m),
            ))

    df = pd.DataFrame(rows)

    close_diff_event_indices = np.where(any_close_mismatch_by_event)[0].tolist()
    return df, exact_fail, close_fail, common_ch, close_diff_event_indices, n_events


# def compare_time_coord_pairs(label_a: str, da_a: xr.DataArray, label_b: str, da_b: xr.DataArray):
#     """
#     Compare the *time coordinate values* between two EEG DataArrays.

#     Assumes:
#       - da_* has a 'time' coordinate or dimension
#       - comparison is by index (crop to min length), not by alignment

#     Returns:
#       df_time_stats : pd.DataFrame with a single row of summary stats
#       exact_fail    : bool (True if any exact mismatch)
#       close_fail    : bool (True if any allclose mismatch)
#       diff_vec      : np.ndarray of time differences (time_a - time_b) on compared indices (NaN where invalid)
#     """
#     if "time" not in da_a.dims or "time" not in da_b.dims:
#         raise ValueError(f"Both inputs must have a 'time' dimension. Got da_a.dims={da_a.dims}, da_b.dims={da_b.dims}")

#     # Pull time coordinate arrays if present; else fall back to index.
#     # (Usually time is a coordinate of the 'time' dim.)
#     if "time" in da_a.coords:
#         t_a = np.asarray(da_a["time"].values)
#     else:
#         t_a = np.arange(da_a.sizes["time"], dtype=float)

#     if "time" in da_b.coords:
#         t_b = np.asarray(da_b["time"].values)
#     else:
#         t_b = np.arange(da_b.sizes["time"], dtype=float)

#     # Ensure 1D
#     t_a = np.squeeze(t_a)
#     t_b = np.squeeze(t_b)

#     # Crop by index to common length
#     t_a2, t_b2, m = _crop_time_to_min(t_a, t_b)

#     # NaN-safe exact + close
#     both_nan = np.isnan(t_a2) & np.isnan(t_b2)
#     exact_bad = ~((t_a2 == t_b2) | both_nan)
#     close_bad = ~np.isclose(t_a2, t_b2, rtol=RTOL, atol=ATOL, equal_nan=True)

#     diff = t_a2 - t_b2
#     invalid = both_nan | np.isnan(t_a2) | np.isnan(t_b2)
#     diff = np.where(invalid, np.nan, diff)
#     abs_diff = np.abs(diff)

#     n_exact = int(np.sum(exact_bad))
#     n_close = int(np.sum(close_bad))

#     mean_abs = float(np.nanmean(abs_diff)) if np.isfinite(abs_diff).any() else np.nan
#     max_abs  = float(np.nanmax(abs_diff))  if np.isfinite(abs_diff).any() else np.nan
#     mean_signed = float(np.nanmean(diff))  if np.isfinite(diff).any() else np.nan
#     std_diff = float(np.nanstd(diff))      if np.isfinite(diff).any() else np.nan
#     mse_time = float(np.nanmean(diff**2))  if np.isfinite(diff).any() else np.nan

#     df = pd.DataFrame([dict(
#         comparison=f"{label_a} vs {label_b}",
#         time_len_a=int(len(t_a)),
#         time_len_b=int(len(t_b)),
#         time_compared_samples=int(m),

#         n_exact_time_diff=n_exact,
#         n_close_time_diff=n_close,

#         mean_abs_time_diff=mean_abs,
#         max_abs_time_diff=max_abs,
#         mean_signed_time_diff=mean_signed,
#         std_time_diff=std_diff,
#         mse_time=mse_time,
#     )])

#     exact_fail = (n_exact != 0)
#     close_fail = (n_close != 0)

#     return df, exact_fail, close_fail, diff

def compare_time_coord_pairs(label_a: str, da_a: xr.DataArray, label_b: str, da_b: xr.DataArray):
    """
    Event-aware comparison of time coordinate values.

    Returns:
      df_time_stats (single row per pair, but includes event mismatch counts/indices),
      exact_fail (bool),
      close_fail (bool),
      diff (np.ndarray of shape (E, T) after crop; NaN where invalid),
      close_diff_event_indices (list[int]),
      n_events (int)
    """
    if "time" not in da_a.dims or "time" not in da_b.dims:
        raise ValueError(f"Both inputs must have a 'time' dimension. Got da_a.dims={da_a.dims}, da_b.dims={da_b.dims}")

    # pull coords (fallback to index)
    t_a = np.asarray(da_a["time"].values) if "time" in da_a.coords else np.arange(da_a.sizes["time"], dtype=float)
    t_b = np.asarray(da_b["time"].values) if "time" in da_b.coords else np.arange(da_b.sizes["time"], dtype=float)

    # normalize to (event, time)
    n_events = int(min(da_a.sizes.get("event", 1), da_b.sizes.get("event", 1)))

    def _to_event_time(t):
        t = np.asarray(t)
        if t.ndim == 1:
            return np.tile(t[None, :], (n_events, 1))
        if t.ndim == 2:
            return t[:n_events]
        raise ValueError(f"Unsupported time coord ndim={t.ndim}; expected 1 or 2.")

    Ta = _to_event_time(t_a)
    Tb = _to_event_time(t_b)

    # crop time to min length
    m = min(Ta.shape[1], Tb.shape[1])
    Ta = Ta[:, :m]
    Tb = Tb[:, :m]

    both_nan = np.isnan(Ta) & np.isnan(Tb)
    exact_bad = ~((Ta == Tb) | both_nan)
    close_bad = ~np.isclose(Ta, Tb, rtol=RTOL, atol=ATOL, equal_nan=True)

    diff = Ta - Tb
    invalid = both_nan | np.isnan(Ta) | np.isnan(Tb)
    diff = np.where(invalid, np.nan, diff)
    abs_diff = np.abs(diff)

    n_exact = int(np.sum(exact_bad))
    n_close = int(np.sum(close_bad))

    # event-level "does this event differ at all?"
    close_bad_by_event = np.any(close_bad, axis=1)
    close_diff_event_indices = np.where(close_bad_by_event)[0].tolist()

    df = pd.DataFrame([dict(
        comparison=f"{label_a} vs {label_b}",
        time_len_a=int(da_a.sizes["time"]),
        time_len_b=int(da_b.sizes["time"]),
        time_compared_samples=int(m),

        n_events=int(n_events),
        n_close_diff_events=int(close_bad_by_event.sum()),
        close_diff_event_indices=close_diff_event_indices,

        n_exact_time_diff=n_exact,
        n_close_time_diff=n_close,

        mean_abs_time_diff=float(np.nanmean(abs_diff)) if np.isfinite(abs_diff).any() else np.nan,
        max_abs_time_diff=float(np.nanmax(abs_diff)) if np.isfinite(abs_diff).any() else np.nan,
        mean_signed_time_diff=float(np.nanmean(diff)) if np.isfinite(diff).any() else np.nan,
        std_time_diff=float(np.nanstd(diff)) if np.isfinite(diff).any() else np.nan,
        mse_time=float(np.nanmean(diff**2)) if np.isfinite(diff).any() else np.nan,
    )])

    exact_fail = (n_exact != 0)
    close_fail = (n_close != 0)

    return df, exact_fail, close_fail, diff, close_diff_event_indices, n_events




def compare_eeg_sources(
    eeg_dict,
    *,
    subject=None,
    experiment=None,
    session=None,
    options=[
        "strip_metadata",
        "print_channel_coord_summary",
        "compare_raw_signals",
        "print_raw_stats",
        "compare_time_coords",
        "print_time_stats",
        "print_mismatched_channels",
        "print_raw_data",
    ],
    channel_dim="channel",
):
    """
    Compare N EEG sources WITHOUT xarray alignment.

    Behavior:
      - Standardizes dim order (event, channel, time)
      - Strips event metadata (optional)
      - Reports channel overlap (pairwise)
      - Pairwise comparisons:
          * channels: intersection by name
          * time: compare by sample index (crop to min length)
    """
    if len(options) <= 0:
        print("no options selected")

    names = list(eeg_dict.keys())
    eegs = list(eeg_dict.values())
    if len(eegs) < 2:
        raise ValueError("Need at least 2 sources to compare.")

    options_set = set(options)

    # ----------------------------
    # Standardize dims (+ optional metadata strip)
    # ----------------------------
    eegs_std = []
    strip_metadata = "strip_metadata" in options_set
    for da in eegs:
        stripped_da = strip_event_metadata(da) if strip_metadata else da
        eegs_std.append(_ensure_dims(stripped_da))

    # ----------------------------
    # Channel summaries (no alignment)
    # ----------------------------
    if "print_channel_coord_summary" in options_set:
        print("\n================ CHANNEL SUMMARY (pre-compare; no alignment) ================")
        for i in range(len(eegs_std)):
            for j in range(i + 1, len(eegs_std)):
                summarize_channel_coords(eegs_std[i], eegs_std[j], channel_dim)

    # Outputs
    df_stats_raw = None
    df_stats_time = None
    df_summary_raw = None  # <-- NEW: DataFrame summary over channels per pair

    # helper for stable nan summaries
    def _nanmean(x):
        x = pd.to_numeric(x, errors="coerce")
        return float(np.nanmean(x)) if np.isfinite(x).any() else np.nan

    def _nanmax(x):
        x = pd.to_numeric(x, errors="coerce")
        return float(np.nanmax(x)) if np.isfinite(x).any() else np.nan

    # ----------------------------
    # Pairwise comparisons: RAW SIGNALS
    # ----------------------------
    if "compare_raw_signals" in options_set:
        print("\n================ PAIRWISE COMPARISONS RAW SIGNAL ================")
        per_channel_frames = []
        summary_rows = []

        for i in range(len(eegs_std)):
            for j in range(i + 1, len(eegs_std)):
                a_name = names[i]
                b_name = names[j]

                # df_pair, exact_fail, close_fail, common_ch = compare_raw_signal_pairs(
                #     a_name, eegs_std[i], b_name, eegs_std[j]
                # )
                df_pair, exact_fail, close_fail, common_ch, close_diff_event_indices, n_events = compare_raw_signal_pairs(
                    a_name, eegs_std[i], b_name, eegs_std[j]
                )

                per_channel_frames.append(df_pair)

                # samples compared = min time length (since compare_raw_signal_pairs crops)
                ta = int(eegs_std[i].sizes.get("time", 0))
                tb = int(eegs_std[j].sizes.get("time", 0))
                n_samples_compared = int(min(ta, tb))
                print(
                    f"{a_name} vs {b_name}: time lengths {ta} vs {tb} "
                    f"(comparing first {n_samples_compared} samples)"
                )

                # Aggregate across channels for this pair
                summary_rows.append(
                    dict(
                        # session identifiers (requested)
                        subject=subject,
                        experiment=experiment,
                        session=session,

                        comparison=f"{a_name} vs {b_name}",
                        source_a=a_name,
                        source_b=b_name,

                        n_common_channels=int(len(common_ch)),
                        n_exact_diff_channels=int(len(exact_fail)),
                        n_close_diff_channels=int(len(close_fail)),

                        # aggregate metrics across channels (means/maxes of per-channel metrics)
                        mean_abs_diff=_nanmean(df_pair["mean_abs_diff"]) if len(df_pair) else np.nan,
                        max_abs_diff=_nanmax(df_pair["max_abs_diff"]) if len(df_pair) else np.nan,
                        mean_signed_diff=_nanmean(df_pair["mean_signed_diff"]) if len(df_pair) else np.nan,
                        std_diff=_nanmean(df_pair["std_diff"]) if len(df_pair) else np.nan,
                        mse=_nanmean(df_pair["mse_channel"]) if len(df_pair) else np.nan,

                        n_samples_compared=n_samples_compared,

                        exact_diff_channels=list(exact_fail),
                        close_diff_channels=list(close_fail),
                        n_events=int(n_events),
                        n_close_diff_events=int(len(close_diff_event_indices)),
                        close_diff_event_indices=list(close_diff_event_indices),

                    )
                )

        df_stats_raw = (
            pd.concat(per_channel_frames, axis=0, ignore_index=True)
            if per_channel_frames else pd.DataFrame()
        )

        # add subject/session/experiment columns to per-channel df too (requested)
        if len(df_stats_raw) > 0:
            df_stats_raw.insert(0, "session", session)
            df_stats_raw.insert(0, "experiment", experiment)
            df_stats_raw.insert(0, "subject", subject)

        df_summary_raw = pd.DataFrame(summary_rows)

        pd.set_option("display.max_rows", None)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 200)

        if "print_raw_stats" in options_set:
            print("\n================ PER-CHANNEL STATS (RAW SIGNAL; ALL PAIRS) ================")
            if len(df_stats_raw) == 0:
                print("[WARN] No raw-signal stats produced (no common channels across any pairs?)")
            else:
                print(df_stats_raw.to_string(index=False))

        if "print_mismatched_channels" in options_set:
            print("\n================ SUMMARY (RAW SIGNAL; aggregated) ================")
            if len(df_summary_raw) == 0:
                print("[WARN] No raw-signal summary produced.")
            else:
                print(df_summary_raw.to_string(index=False))

    # ----------------------------
    # Pairwise comparisons: TIME COORDS
    # ----------------------------
    if "compare_time_coords" in options_set:
        print("\n================ PAIRWISE COMPARISONS TIME COORDS ================")
        time_frames = []

        for i in range(len(eegs_std)):
            for j in range(i + 1, len(eegs_std)):
                a_name = names[i]
                b_name = names[j]
                # df, exact_fail, close_fail, diff, close_diff_event_indices, n_events
                df_time, exact_fail, close_fail, diff_vec, close_diff_event_indices, n_events = compare_time_coord_pairs(
                    a_name, eegs_std[i], b_name, eegs_std[j]
                )
                time_frames.append(df_time)

                ta = int(eegs_std[i].sizes.get("time", 0))
                tb = int(eegs_std[j].sizes.get("time", 0))
                print(
                    f"{a_name} vs {b_name}: time lengths {ta} vs {tb} "
                    f"(comparing first {min(ta, tb)} time coords)"
                )

        df_stats_time = (
            pd.concat(time_frames, axis=0, ignore_index=True)
            if time_frames else pd.DataFrame()
        )

        # add subject/session/experiment columns to time df too (requested)
        if len(df_stats_time) > 0:
            df_stats_time.insert(0, "session", session)
            df_stats_time.insert(0, "experiment", experiment)
            df_stats_time.insert(0, "subject", subject)

        pd.set_option("display.max_rows", None)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 200)

        if "print_time_stats" in options_set:
            print("\n================ TIME COORD STATS (ALL PAIRS) ================")
            if len(df_stats_time) == 0:
                print("[WARN] No time-coord stats produced.")
            else:
                print(df_stats_time.to_string(index=False))

    # ----------------------------
    # Optional raw prints
    # ----------------------------
    if "print_raw_data" in options_set:
        for nm, da in zip(names, eegs_std):
            print_raw_data(nm, da)

    # NOTE: summary_time removed (you said it duplicates df_time)
    return {
        "df_raw": df_stats_raw,
        "df_raw_summary": df_summary_raw,  # <-- DataFrame summary for raw signals
        "df_time": df_stats_time,
        # "eegs_std": eegs_std,
    }


# Convert to DataFrame
def fix_evs_bids(full_evs):
    value_recalls = full_evs[full_evs.trial_type == "VALUE_RECALL"] 
    words = full_evs[full_evs.trial_type == "WORD"]
    rec_words = full_evs[full_evs.trial_type == "REC_WORD"]
    rec_vv_words = full_evs[full_evs.trial_type == "REC_WORD_VV"]

    # WORD --> storepointtype, recalled--> VALUE_RECALL, REC_WORD, REC_WORD_VV
    word_trial_to_storepointtype = words.set_index("trial")["storepointtype"].to_dict()
    word_trial_to_recalled = words.set_index("trial")["recalled"].to_dict()
    for event_type in ["VALUE_RECALL", "REC_WORD", "REC_WORD_VV"]:
        subset = full_evs[full_evs.trial_type == event_type]
        for idx, row in subset.iterrows():
            trial = row["trial"]
            if trial in word_trial_to_storepointtype:
                full_evs.at[idx, "storepointtype"] = word_trial_to_storepointtype[trial]
            if trial in word_trial_to_recalled:
                full_evs.at[idx, "recalled"] = word_trial_to_recalled[trial]

    # VALUE_RECALL --> actualvalue, valuerecall --> WORD, `REC_WORD`, REC_WORD_VV
    valuerecall_trial_to_actualvalue = value_recalls.set_index("trial")["actualvalue"].to_dict()
    valuerecall_trial_to_valuerecall = value_recalls.set_index("trial")["valuerecall"].to_dict()

    # --- Apply to multi-row event types ---
    for event_type in ["WORD", "REC_WORD", "REC_WORD_VV"]:
        subset = full_evs[full_evs.trial_type == event_type]
        for idx, row in subset.iterrows():
            trial = row["trial"]

            # actualvalue
            if trial in valuerecall_trial_to_actualvalue:
                full_evs.at[idx, "actualvalue"] = valuerecall_trial_to_actualvalue[trial]

            # valuerecall
            if trial in valuerecall_trial_to_valuerecall:
                full_evs.at[idx, "valuerecall"] = valuerecall_trial_to_valuerecall[trial]
                
    return full_evs

def fix_evs_cml(full_evs):
    value_recalls = full_evs[full_evs.type == "VALUE_RECALL"] 
    words = full_evs[full_evs.type == "WORD"]
    rec_words = full_evs[full_evs.type == "REC_WORD"]
    rec_vv_words = full_evs[full_evs.type == "REC_WORD_VV"]

    # WORD --> storepointtype, recalled--> VALUE_RECALL, REC_WORD, REC_WORD_VV
    word_trial_to_storepointtype = words.set_index("trial")["storepointtype"].to_dict()
    word_trial_to_recalled = words.set_index("trial")["recalled"].to_dict()
    for event_type in ["VALUE_RECALL", "REC_WORD", "REC_WORD_VV"]:
        subset = full_evs[full_evs.type == event_type]
        for idx, row in subset.iterrows():
            trial = row["trial"]
            if trial in word_trial_to_storepointtype:
                full_evs.at[idx, "storepointtype"] = word_trial_to_storepointtype[trial]
            if trial in word_trial_to_recalled:
                full_evs.at[idx, "recalled"] = word_trial_to_recalled[trial]

    # VALUE_RECALL --> actualvalue, valuerecall --> WORD, `REC_WORD`, REC_WORD_VV
    valuerecall_trial_to_actualvalue = value_recalls.set_index("trial")["actualvalue"].to_dict()
    valuerecall_trial_to_valuerecall = value_recalls.set_index("trial")["valuerecall"].to_dict()

    # --- Apply to multi-row event types ---
    for event_type in ["WORD", "REC_WORD", "REC_WORD_VV"]:
        subset = full_evs[full_evs.type == event_type]
        for idx, row in subset.iterrows():
            trial = row["trial"]

            # actualvalue
            if trial in valuerecall_trial_to_actualvalue:
                full_evs.at[idx, "actualvalue"] = valuerecall_trial_to_actualvalue[trial]

            # valuerecall
            if trial in valuerecall_trial_to_valuerecall:
                full_evs.at[idx, "valuerecall"] = valuerecall_trial_to_valuerecall[trial]
                
    return full_evs



# subject
def process_raw_signals(sub, exp, sess, bids_root, out_path): # entire signal, not epoched
    ### load cml
    reader = cml.CMLReader(subject=sub, experiment=exp, session=sess)
    eeg_cml = reader.load_eeg().to_ptsa()

    ### load bdf
    # BIDS
    bids_path = BIDSPath(
        subject=sub,
        session=str(sess),
        task=exp.lower(),
        datatype="eeg",
        root=bids_root,
    )

    raw = read_raw_bids(
        bids_path,
        verbose=True,
    )

    eeg_bids = xr.DataArray(
        raw.get_data()[None, :, :],                           # -> (1, n_channels, n_times)
        dims=("event", "channel", "time"),         # match eeg_cml dim names
        coords={
            "event": [0],                          # singleton event index
            "channel": raw.ch_names,
            "time": raw.times * 1000,
            "samplerate": raw.info["sfreq"],                    # scalar coord (optional)
        },
        name="eeg",
    )

    ## load pyedf
    # cml_bdf_path  = f"/protocols/ltp/subjects/{sub}/experiments/{exp}/sessions/{sess}/ephys/current_processed/{sub}_session_{sess}.bdf"
    # eeg_pyedflib = load_bdf_as_xarray(cml_bdf_path)

    # compare
    results = compare_eeg_sources(
        eeg_dict={"BIDS": eeg_bids, "CMLReader": eeg_cml},
        subject=sub,
        experiment=exp,
        session=sess,
        options=["strip_metadata", "compare_raw_signals", "compare_time_coords"]
    )
    
    results["df_raw"].to_csv(f"{out_path}df_raw_{sub}_{exp}_{sess}.csv", index=False)
    results["df_raw_summary"].to_csv(f"{out_path}df_raw_summary_{sub}_{exp}_{sess}.csv", index=False)
    results["df_time"].to_csv(f"{out_path}df_time_{sub}_{exp}_{sess}.csv", index=False)
    return results

def _all_exist(paths):
    return all(os.path.exists(p) for p in paths)

def _dedupe_events_by_sample(df: pd.DataFrame, sample_col: str, *, keep="first") -> pd.DataFrame:
    if sample_col not in df.columns:
        raise ValueError(f"Expected column '{sample_col}' in events df. Columns={list(df.columns)[:20]}")
    df2 = df.copy()
    df2[sample_col] = pd.to_numeric(df2[sample_col], errors="coerce")
    df2 = df2.dropna(subset=[sample_col])
    df2 = df2.sort_values(sample_col, kind="mergesort")
    df2 = df2[~df2[sample_col].duplicated(keep=keep)]
    return df2

def _as_list(x):
    if x is None:
        return None
    if isinstance(x, (list, tuple, set, np.ndarray, pd.Index)):
        return list(x)
    return [x]

def process_epoched_signals_by_type(
    sub,
    exp,
    sess,
    evs_types,
    tmin,
    tmax,
    bids_root,
    out_path,
    *,
    skip_if_exists=True,
    keep="first",
    verbose=False,
):
    """
    Run epoch+compare separately for each event type, append results across types,
    save and return the appended DataFrames.
    """
    os.makedirs(out_path, exist_ok=True)

    # aggregated outputs (ONE set per sub/exp/sess)
    out_raw = os.path.join(out_path, f"df_raw_{sub}_{exp}_{sess}.csv")
    out_raw_summary = os.path.join(out_path, f"df_raw_summary_{sub}_{exp}_{sess}.csv")
    out_time = os.path.join(out_path, f"df_time_{sub}_{exp}_{sess}.csv")
    expected = [out_raw, out_raw_summary, out_time]

    if skip_if_exists and _all_exist(expected):
        print("Files exist: skipped")
        return {"skipped": True, "reason": "outputs_exist", "paths": expected}

    # --------------------------
    # CML: load events once
    # --------------------------
    cmlreader = cml.CMLReader(subject=sub, experiment=exp, session=sess)
    evs_cml = cmlreader.load("events")

    # decide which types to run
    if evs_types is None:
        types_to_run = sorted(pd.unique(evs_cml["type"]))
    else:
        types_to_run = sorted(set(_as_list(evs_types)))

    if len(types_to_run) == 0:
        raise ValueError("types_to_run is empty.")

    # --------------------------
    # BIDS: load raw + annotations once
    # --------------------------
    task = exp.lower()
    bids_path = BIDSPath(
        subject=sub,
        session=str(sess),
        task=task,
        datatype="eeg",
        root=bids_root,
    )

    raw_bids = read_raw_bids(bids_path)
    raw_bids.set_channel_types({
        "EXG1": "eog", "EXG2": "eog", "EXG3": "eog", "EXG4": "eog",
        "EXG5": "misc", "EXG6": "misc", "EXG7": "misc", "EXG8": "misc",
    })

    events_all, event_id_all = mne.events_from_annotations(raw_bids)
    sfreq = float(raw_bids.info["sfreq"])

    # collect per-type outputs
    all_raw = []
    all_raw_summary = []
    all_time = []

    # optional bookkeeping
    per_type_status = []

    for etype in types_to_run:
        if verbose:
            print(f"[{sub} | {exp} | {sess}] type={etype}")

        try:
            # --------------------------
            # CML: filter to this type + dedupe by eegoffset, then epoch
            # --------------------------
            evs_cml_t = evs_cml[evs_cml["type"] == etype].copy()
            if evs_cml_t.empty:
                per_type_status.append((etype, "skip", "no_cml_events"))
                continue

            evs_cml_t = _dedupe_events_by_sample(evs_cml_t, "eegoffset", keep=keep)

            eeg_cml = cmlreader.load_eeg(evs_cml_t, rel_start=tmin, rel_stop=tmax).to_ptsa()

            # --------------------------
            # BIDS: filter annotation labels/codes for this type, dedupe by sample, epoch
            # --------------------------
            if etype not in event_id_all:
                per_type_status.append((etype, "skip", "etype_not_in_annotations"))
                # free CML epoch before continue
                del eeg_cml
                gc.collect()
                continue

            filtered_event_id = {etype: event_id_all[etype]}
            code = filtered_event_id[etype]

            events_filt = events_all[events_all[:, 2] == code]
            if len(events_filt) == 0:
                per_type_status.append((etype, "skip", "no_bids_events"))
                del eeg_cml
                gc.collect()
                continue

            # dedupe by sample
            _, first_idx = np.unique(events_filt[:, 0], return_index=True)
            events_filt = events_filt[np.sort(first_idx)]

            epochs_bids = mne.Epochs(
                raw_bids,
                events=events_filt,
                event_id=filtered_event_id,
                tmin=tmin / 1000.0,
                tmax=tmax / 1000.0,
                baseline=None,
                preload=True,
            )

            picks_eeg = mne.pick_types(epochs_bids.info, eeg=True, eog=False, misc=False)
            epochs_bids = epochs_bids.pick(picks_eeg)

            # metadata aligned to events_filt
            meta = pd.DataFrame({
                "sample": events_filt[:, 0].astype(int),
                "trial_type": [etype] * len(events_filt),
            })
            meta["onset"] = meta["sample"] / sfreq

            eeg_bids = TimeSeries.from_mne_epochs(epochs_bids, meta)
            eeg_bids = eeg_bids.assign_coords(time=eeg_bids["time"] * 1000.0)
            eeg_bids["time"].attrs["units"] = "ms"

            # --------------------------
            # Compare
            # --------------------------
            res = compare_eeg_sources(
                eeg_dict={"BIDS": eeg_bids, "CMLReader": eeg_cml},
                subject=sub,
                experiment=exp,
                session=sess,
                options=["strip_metadata", "compare_raw_signals", "compare_time_coords"],
            )

            # append dfs; add event type column so you can stratify later
            if res.get("df_raw") is not None and not res["df_raw"].empty:
                df = res["df_raw"].copy()
                df["event_type"] = etype
                all_raw.append(df)

            if res.get("df_raw_summary") is not None and not res["df_raw_summary"].empty:
                df = res["df_raw_summary"].copy()
                df["event_type"] = etype
                all_raw_summary.append(df)

            if res.get("df_time") is not None and not res["df_time"].empty:
                df = res["df_time"].copy()
                df["event_type"] = etype
                all_time.append(df)

            per_type_status.append((etype, "ok", ""))

        except Exception as e:
            per_type_status.append((etype, "fail", repr(e)))

        finally:
            # free big objects per type
            for name in ("epochs_bids", "eeg_bids", "eeg_cml", "res", "events_filt", "meta"):
                if name in locals():
                    try:
                        del locals()[name]
                    except Exception:
                        pass
            gc.collect()

    # done with BIDS raw
    try:
        raw_bids.close()
    except Exception:
        pass
    del raw_bids
    gc.collect()

    # concatenate and save
    df_raw_all = pd.concat(all_raw, ignore_index=True) if all_raw else pd.DataFrame()
    df_raw_summary_all = pd.concat(all_raw_summary, ignore_index=True) if all_raw_summary else pd.DataFrame()
    df_time_all = pd.concat(all_time, ignore_index=True) if all_time else pd.DataFrame()

    df_raw_all.to_csv(out_raw, index=False)
    df_raw_summary_all.to_csv(out_raw_summary, index=False)
    df_time_all.to_csv(out_time, index=False)

    return {
        "df_raw": df_raw_all,
        "df_raw_summary": df_raw_summary_all,
        "df_time": df_time_all,
        "per_type_status": pd.DataFrame(per_type_status, columns=["event_type", "status", "detail"]),
        "paths": expected,
    }

def _all_exist(paths):
    return all(os.path.exists(p) for p in paths)

def load_bids_events(sub, exp, sess, bids_root, *, return_path=False):
    """
    Load BIDS events.tsv trying both ieeg/ and eeg/ folders.

    Tries:
      datatype: ieeg -> eeg
      task variants: exp, exp.lower(), exp.upper()

    Returns:
      df (or (df, path) if return_path=True), or None if not found.
    """
    datatypes = ("ieeg", "eeg")
    task_variants = []
    for t in (exp, str(exp).lower(), str(exp).upper()):
        if t not in task_variants:
            task_variants.append(t)

    for datatype in datatypes:
        for task in task_variants:
            bp = BIDSPath(
                subject=sub,
                session=str(sess),
                task=task,
                datatype=datatype,
                suffix="events",
                extension=".tsv",
                root=bids_root,
                check=False,
            )
            fpath = bp.fpath
            if fpath is not None and os.path.exists(fpath):
                df = pd.read_csv(fpath, sep="\t")
                return (df, fpath) if return_path else df

    return None

def process_events(sub, exp, sess, evs_types, bids_root, out_path, *, skip_if_exists=True):
    os.makedirs(out_path, exist_ok=True)
    out_behavior_summary = os.path.join(out_path, f"df_behavior_summary_{sub}_{exp}_{sess}.csv")
    
    expected = [out_behavior_summary]
    if skip_if_exists and _all_exist(expected):
        return {"skipped": True, "reason": "outputs_exist", "paths": expected}
    
    # Load CML events
    try:
        cmlreader = cml.CMLReader(subject=sub, experiment=exp, session=sess)
        evs_cml = cmlreader.load('events')
        # print(evs_cml.columns)
        # print(evs_cml)
    except Exception as e:
        print(f"Failed to load CML events for {sub} | {exp} | {sess}: {e}")
        return {"skipped": True, "reason": "cml_load_failed", "error": str(e)}
    
    evs_types_set = set(evs_types) if evs_types is not None else set(evs_cml["type"].unique())
    # print(evs_cml.columns)
    if exp == "ValueCourier":
        evs_cml = fix_evs_cml(evs_cml)
    # print(evs_cml.columns)
    
    filtered_evs_cml = evs_cml[evs_cml["type"].isin(evs_types_set)]
    # print(filtered_evs_cml.columns)
    
    # Load BIDS events
    tmp = load_bids_events(sub, exp, sess, bids_root, return_path=True)
    if tmp is None:
        print(f"Skipping {sub} | {exp} | {sess}: BIDS events file not found in eeg/ or ieeg/")
        return {"skipped": True, "reason": "bids_events_not_found"}

    evs_bids, bids_events_path = tmp
    print(f"[BIDS] Loaded events from: {bids_events_path}")
    
    if evs_bids is None:
        print(f"Skipping {sub} | {exp} | {sess}: BIDS events file not found")
        return {"skipped": True, "reason": "bids_events_not_found"}
    
    if exp == "ValueCourier":
        evs_bids = fix_evs_bids(evs_bids)
    
    
    # Check for required columns
    required_cols = {'sample', 'onset', 'trial_type'}
    missing_cols = required_cols - set(evs_bids.columns)
    if missing_cols:
        print(f"Skipping {sub} | {exp} | {sess}: BIDS events missing required columns: {missing_cols}")
        print(f"Available columns: {list(evs_bids.columns)}")
        return {"skipped": True, "reason": "missing_columns", "missing": list(missing_cols)}
    
    filtered_evs_bids = evs_bids[evs_bids["trial_type"].isin(evs_types_set)]
    
    if filtered_evs_bids.empty:
        print(f"Skipping {sub} | {exp} | {sess}: No events match the requested types")
        return {"skipped": True, "reason": "no_matching_events"}
    print(filtered_evs_bids.columns)
    # Compare behavioral data
    try:
        i = 0
        results = compare_behavioral(
            filtered_evs_cml, "CMLReader",
            filtered_evs_bids, "OpenBIDS",
            options=[
                "compare_onset_as_diff",
                "tolerant_numeric",
                "return_col_summary",
                "return_mismatches",
            ],
            drop_cols=[],
        )
        
        os.makedirs(out_path, exist_ok=True)
        results["df_behavior_summary"].to_csv(
            os.path.join(out_path, f"df_behavior_summary_{sub}_{exp}_{sess}.csv"),
            index=False,
        )
        
        print(f"Successfully processed {sub} | {exp} | {sess}")
        return results
        
    except Exception as e:
        print(f"Failed to compare events for {sub} | {exp} | {sess}: {e}")
        import traceback
        traceback.print_exc()
        return {"skipped": True, "reason": "comparison_failed", "error": str(e)}
    


def load_and_concat(file_list, remove_duplicates=True):
    """
    Load and concatenate CSV files with duplicate handling.
    
    Parameters:
    -----------
    file_list : list
        List of file paths to concatenate
    remove_duplicates : bool
        Whether to remove duplicate rows (default: True)
    
    Returns:
    --------
    pd.DataFrame
        Concatenated DataFrame with duplicates optionally removed
    """
    if not file_list:
        return pd.DataFrame()  # Return empty DF if no files found
    
    # Read each CSV and combine them into one, skipping empty files
    dfs = []
    for f in file_list:
        try:
            df = pd.read_csv(f)
            if not df.empty:
                dfs.append(df)
            else:
                print(f"Warning: Skipping empty file: {f}")
        except pd.errors.EmptyDataError:
            print(f"Warning: Skipping empty/malformed file: {f}")
        except Exception as e:
            print(f"Warning: Error reading {f}: {e}")
    
    if not dfs:
        print("Warning: No valid CSV files found to concatenate")
        return pd.DataFrame()
    
    df = pd.concat(dfs, ignore_index=True)
    
    if remove_duplicates:
        initial_rows = len(df)
        df = df.drop_duplicates()
        removed_rows = initial_rows - len(df)
        if removed_rows > 0:
            print(f"Removed {removed_rows} duplicate rows")
    
    return df

def delete_source_files(file_list, delete_files=False):
    """
    Delete source files after successful concatenation.
    
    Parameters:
    -----------
    file_list : list
        List of file paths to delete
    delete_files : bool
        Whether to actually delete the files (default: False for safety)
    """
    if delete_files and file_list:
        for f in file_list:
            try:
                os.remove(f)
                print(f"Deleted: {f}")
            except Exception as e:
                print(f"Error deleting {f}: {e}")
    
### PLOTTING
def plot_comp_results(df_results, col_tgt, col_std=None, col_label=None):
    # plot mean and std difference
    comparisons = df_results['comparison'].unique()
    subjects = df_results['subject'].unique()
    experiments = df_results['experiment'].unique()

    for experiment in experiments:
        fig, axes = plt.subplots(1, len(comparisons), figsize=(6 * len(comparisons), 5), sharex=True)
        if len(comparisons) == 1: axes = [axes]

        for i, comp in enumerate(comparisons):
            ax = axes[i]
            comp_df = df_results[(df_results['comparison'] == comp) & (df_results['experiment'] == experiment)]

            for subj in subjects:
                subj_df = comp_df[comp_df['subject'] == subj].sort_values('session')
                if subj_df.empty: continue

                # Plot mean line
                line, = ax.plot(subj_df['session'], subj_df[col_tgt], marker='o', label=subj)

                # Add shaded Std region
                if col_std is not None:
                    ax.fill_between(
                        subj_df['session'], 
                        subj_df[col_tgt] - subj_df[col_std],
                        subj_df[col_tgt] + subj_df[col_std],
                        color=line.get_color(), 
                        alpha=0.15
                    )

            ax.set_title(f"{experiment} | {comp}")
            ax.set_xlabel('Session')
            y_label = col_label if col_label is not None else col_tgt
            y_label += " ($\pm$ Std)" if col_std is not None else ""
            if i == 0: ax.set_ylabel(y_label)
            ax.legend(title='Subject')

        plt.tight_layout()
        plt.show()