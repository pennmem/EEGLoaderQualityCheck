"""Normalize CML and BIDS event DataFrames into a shared schema."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Sequence, Set, Union

import numpy as np
import pandas as pd

from ..comparators.utils import one_unique


def prep_events(
    evs_cml: pd.DataFrame,
    evs_bids: pd.DataFrame,
    *,
    evs_types: Optional[Sequence[str]] = None,
    onset_as_diff: bool = True,
    drop_cols: Iterable[str] = (),
    subject: Optional[str] = None,
    experiment: Optional[str] = None,
    session: Optional[Union[str, int]] = None,
) -> Dict[str, Any]:
    """Align CML and BIDS events into comparable DataFrames.

    Normalization steps:
    - CML: ``eegoffset`` → ``sample``, ``mstime`` → ``onset`` (sec), ``type`` → ``trial_type``
    - Filters both to requested ``evs_types``
    - Replaces CML sentinels (-999, "") with NaN
    - Onset: either ``diff(onset)`` or shift CML to start at 0

    Returns dict with ``evs_cml_prepped``, ``evs_bids_prepped``, metadata.
    """
    drop_cols = set(drop_cols or ())

    # Infer metadata
    subject = subject or one_unique(evs_cml, "subject") or one_unique(evs_bids, "subject")
    experiment = experiment or one_unique(evs_cml, "experiment") or one_unique(evs_bids, "experiment")
    session = session or one_unique(evs_cml, "session") or one_unique(evs_bids, "session")

    # Validate required columns
    for col in ("eegoffset", "mstime", "type"):
        if col not in evs_cml.columns:
            raise ValueError(f"CML: missing required column '{col}'")
    for col in ("sample", "onset", "trial_type"):
        if col not in evs_bids.columns:
            raise ValueError(f"BIDS: missing required column '{col}'")

    # Event-type filtering
    if evs_types is None:
        evs_types_used = set(evs_cml["type"].dropna().astype(str).unique())
    else:
        evs_types_used = set(map(str, evs_types))

    cml_f = evs_cml[evs_cml["type"].astype(str).isin(evs_types_used)].copy()
    bids_f = evs_bids[evs_bids["trial_type"].astype(str).isin(evs_types_used)].copy()

    # Replace sentinels on both sides so missing/placeholder values compare equal.
    sentinels = {-999: np.nan, -999.0: np.nan, "-999": np.nan,
                 "": np.nan, "n/a": np.nan, "N/A": np.nan, "X": np.nan}
    cml2 = cml_f.replace(sentinels).copy()
    bids2 = bids_f.replace(sentinels).copy()

    # Rename CML columns to BIDS schema
    cml2 = cml2.rename(columns={"eegoffset": "sample", "mstime": "onset", "type": "trial_type"})

    # Ensure comparable types
    cml2["sample"] = pd.to_numeric(cml2["sample"], errors="coerce")
    bids2["sample"] = pd.to_numeric(bids2["sample"], errors="coerce")
    cml2["trial_type"] = cml2["trial_type"].astype(str)
    bids2["trial_type"] = bids2["trial_type"].astype(str)

    # CML stores list-valued columns (e.g. `test`) as Python lists, while BIDS
    # round-trips them through TSV as strings. Stringify list cells on the CML
    # side so the two compare equal; preserve NaN.
    for col in cml2.columns:
        if cml2[col].apply(lambda v: isinstance(v, list)).any():
            cml2[col] = cml2[col].apply(lambda v: str(v) if isinstance(v, list) else v)

    # Onset conversion. Anchor the zero-point at the first event in the
    # *unfiltered* frame so CML and BIDS share an absolute timeline even
    # when evs_types drops the original first row.
    cml_onset_s = pd.to_numeric(cml_f["mstime"], errors="coerce") / 1000.0
    bids_onset_s = pd.to_numeric(bids2["onset"], errors="coerce")
    cml_t0 = pd.to_numeric(evs_cml["mstime"], errors="coerce").iloc[0] / 1000.0
    bids_t0 = pd.to_numeric(evs_bids["onset"], errors="coerce").iloc[0]

    if onset_as_diff:
        cml2["onset"] = cml_onset_s.diff()
        bids2["onset"] = bids_onset_s.diff()
    else:
        cml2["onset"] = cml_onset_s - cml_t0
        bids2["onset"] = bids_onset_s - bids_t0

    # Attach metadata
    for df in (cml2, bids2):
        if "subject" not in df.columns:
            df["subject"] = subject
        if "experiment" not in df.columns:
            df["experiment"] = experiment
        if "session" not in df.columns:
            df["session"] = session

    # Drop columns
    if drop_cols:
        cml2 = cml2.drop(columns=[c for c in drop_cols if c in cml2.columns])
        bids2 = bids2.drop(columns=[c for c in drop_cols if c in bids2.columns])

    # # Sort both sides by sample then trial_type so downstream alignment is stable.
    # cml2 = cml2.sort_values(["sample", "trial_type"], kind="mergesort")
    # bids2 = bids2.sort_values(["sample", "trial_type"], kind="mergesort")

    return {
        "evs_cml_prepped": cml2.reset_index(drop=True),
        "evs_bids_prepped": bids2.reset_index(drop=True),
        "subject": subject,
        "experiment": experiment,
        "session": session,
        "evs_types_used": sorted(evs_types_used),
    }


def dedupe_events_by_sample(
    df: pd.DataFrame,
    sample_col: str = "eegoffset",
    *,
    keep: str = "first",
) -> pd.DataFrame:
    """Drop duplicate events sharing the same sample offset."""
    df2 = df.copy()
    df2[sample_col] = pd.to_numeric(df2[sample_col], errors="coerce")
    df2 = df2.dropna(subset=[sample_col])
    df2 = df2.sort_values(sample_col, kind="mergesort")
    return df2[~df2[sample_col].duplicated(keep=keep)]
