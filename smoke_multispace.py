"""Smoke test: multi-space montage comparison.

Runs the updated MontagePipeline on one FR1 session and prints enough
of each output CSV to verify:
  * One row per (session, space) in df_{contacts,pairs}_summary
  * `space` column present; redundant columns absent
  * df_montage_status carries the `space` column

Run:  python smoke_multispace.py
"""
from __future__ import annotations

import os
import sys

import pandas as pd

# Point at the eeg-validation checkout and the local bidsreader so the
# checkout wins over anything in site-packages.
_EEG_VAL = "/home1/zrentala/eeg-validation"
if _EEG_VAL not in sys.path:
    sys.path.insert(0, _EEG_VAL)

# Must be insert(0, ...), not append: a site-packages install of
# bidsreader would otherwise win and your local checkout never loads.
sys.path.insert(0, "/home1/zrentala/bidsreader")
# Evict any already-imported bidsreader (e.g. a pre-existing installed
# copy pulled in by an earlier import) so our checkout wins.
for _mod in [m for m in list(sys.modules) if m == "bidsreader" or m.startswith("bidsreader.")]:
    del sys.modules[_mod]
from bidsreader import CMLBIDSReader  # noqa: E402,F401
print(f"Using bidsreader at: {sys.modules['bidsreader'].__file__}")

from eeg_validation.pipelines.montage import MontagePipeline  # noqa: E402

SUBJECT = "R1001P"
EXPERIMENT = "FR1"
SESSION = 1
BIDS_ROOT = "/data/LTP_BIDS/pyedflib/FR1/"
OUT_DIR = "/home1/zrentala/eeg-validation/results/smoke_test"

DROPPED_COLS = {
    "source_a", "source_b", "align_mode", "sort_keys_used",
    "n_differing_columns", "tolerant_numeric", "any_mismatch",
}


def _assert_schema(df: pd.DataFrame, name: str, *, must_have_space: bool = True) -> None:
    cols = set(df.columns)
    bad = cols & DROPPED_COLS
    assert not bad, f"{name}: unexpected columns still present: {bad}"
    if must_have_space:
        assert "space" in cols, f"{name}: missing 'space' column (cols={sorted(cols)})"
    space_vals = sorted(df["space"].dropna().unique()) if "space" in cols else "n/a"
    print(f"  [OK] {name} — rows={len(df)}, space_values={space_vals}")


def _run(acq: str, out_dir: str) -> None:
    print(f"\n=== acquisition={acq} ===")
    pipe = MontagePipeline(
        subject=SUBJECT,
        experiment=EXPERIMENT,
        session=SESSION,
        bids_root=BIDS_ROOT,
        out_path=out_dir,
        acquisition=acq,
        skip_if_exists=False,
        verbose=True,
    )
    pipe.run()

    tag = f"{SUBJECT}_{EXPERIMENT}_{SESSION}"
    suffix = acq  # 'contacts' or 'pairs'

    summary_p = os.path.join(out_dir, f"df_{suffix}_summary_{tag}.csv")
    detail_p = os.path.join(out_dir, f"df_{suffix}_column_summary_{tag}.csv")
    mism_p = os.path.join(out_dir, f"df_{suffix}_mismatches_{tag}.csv")
    montage_p = os.path.join(out_dir, f"df_montage_status_{acq}_{tag}.csv")

    assert os.path.exists(summary_p), f"missing {summary_p}"
    assert os.path.exists(montage_p), f"missing {montage_p}"

    _assert_schema(pd.read_csv(summary_p), f"{suffix}_summary")
    if os.path.exists(detail_p):
        _assert_schema(pd.read_csv(detail_p), f"{suffix}_column_summary")
    if os.path.exists(mism_p) and os.path.getsize(mism_p) > 0:
        try:
            df_m = pd.read_csv(mism_p)
        except pd.errors.EmptyDataError:
            df_m = None  # header-less empty file = no mismatches, skip
        if df_m is not None and len(df_m) > 0:
            _assert_schema(df_m, f"{suffix}_mismatches")
        else:
            print(f"  [OK] {suffix}_mismatches — empty (no row-level mismatches)")
    _assert_schema(pd.read_csv(montage_p), "montage_summary")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Writing pipeline outputs to: {OUT_DIR}")
    _run("contacts", OUT_DIR)
    _run("pairs", OUT_DIR)
    print("\nALL SMOKE CHECKS PASSED")


if __name__ == "__main__":
    main()
