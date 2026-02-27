"""Experiment-specific event fixes (e.g. ValueCourier field propagation).

The old ``fix_evs_cml`` and ``fix_evs_bids`` were nearly identical —
the only difference was the column used for event type (``type`` vs
``trial_type``).  This module provides a single generic function.
"""

from __future__ import annotations

import pandas as pd


def _propagate_fields(
    full_evs: pd.DataFrame,
    *,
    type_col: str,
    trial_col: str = "trial",
) -> pd.DataFrame:
    """ValueCourier-specific: propagate fields across event types by trial.

    Works for both CML (``type_col='type'``) and BIDS (``type_col='trial_type'``).
    """
    full_evs = full_evs.copy()

    words = full_evs[full_evs[type_col] == "WORD"]
    value_recalls = full_evs[full_evs[type_col] == "VALUE_RECALL"]

    # WORD → storepointtype, recalled → VALUE_RECALL, REC_WORD, REC_WORD_VV
    word_spt = words.set_index(trial_col)["storepointtype"].to_dict()
    word_rec = words.set_index(trial_col)["recalled"].to_dict()

    for etype in ("VALUE_RECALL", "REC_WORD", "REC_WORD_VV"):
        mask = full_evs[type_col] == etype
        trials = full_evs.loc[mask, trial_col]
        full_evs.loc[mask, "storepointtype"] = trials.map(word_spt)
        full_evs.loc[mask, "recalled"] = trials.map(word_rec)

    # VALUE_RECALL → actualvalue, valuerecall → WORD, REC_WORD, REC_WORD_VV
    vr_av = value_recalls.set_index(trial_col)["actualvalue"].to_dict()
    vr_vr = value_recalls.set_index(trial_col)["valuerecall"].to_dict()

    for etype in ("WORD", "REC_WORD", "REC_WORD_VV"):
        mask = full_evs[type_col] == etype
        trials = full_evs.loc[mask, trial_col]
        full_evs.loc[mask, "actualvalue"] = trials.map(vr_av)
        full_evs.loc[mask, "valuerecall"] = trials.map(vr_vr)

    return full_evs


def fix_value_courier_cml(evs: pd.DataFrame) -> pd.DataFrame:
    return _propagate_fields(evs, type_col="type")


def fix_value_courier_bids(evs: pd.DataFrame) -> pd.DataFrame:
    return _propagate_fields(evs, type_col="trial_type")


# Registry: experiment name → (cml_fixer, bids_fixer)
EXPERIMENT_FIXES = {
    "ValueCourier": (fix_value_courier_cml, fix_value_courier_bids),
}


def apply_fixes(
    experiment: str,
    evs_cml: pd.DataFrame,
    evs_bids: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply experiment-specific fixes if registered."""
    if experiment in EXPERIMENT_FIXES:
        cml_fix, bids_fix = EXPERIMENT_FIXES[experiment]
        evs_cml = cml_fix(evs_cml)
        evs_bids = bids_fix(evs_bids)
    return evs_cml, evs_bids
