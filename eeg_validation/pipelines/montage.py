"""Montage (contacts or pairs) comparison pipeline.

Uses BIDSReader.load_electrodes() and BIDSReader.load_channels() instead
of manually constructing BIDSPaths.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from .base import BasePipeline
from ..loaders.cml import load_cml_contacts_and_pairs
from ..loaders.bids import load_bids_channels
from ..preparers.montage import prep_contacts, prep_pairs, BIDS_TO_CML_SPACE
from ..comparators.dataframe import DataFrameComparator


class MontagePipeline(BasePipeline):
    """Compare CML vs BIDS contacts or pairs for one iEEG session.

    Set ``acquisition="contacts"`` to compare contacts (monopolar),
    or ``acquisition="pairs"`` to compare pairs (bipolar).
    """

    def __init__(self, *args, atol_coords: float = 1e-3, rtol_coords: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.atol_coords = atol_coords
        self.rtol_coords = rtol_coords

    def _output_paths(self) -> List[str]:
        tag = self.session_tag
        acq = self.acq_label
        return [self._make_path(f"df_montage_status_{acq}")]

    def _run(self) -> Dict[str, Any]:
        self._vprint(f"  Loading CML contacts and pairs...")
        contacts_cml, pairs_cml = load_cml_contacts_and_pairs(
            self.subject, self.experiment, self.session,
            self.localization, self.montage,
        )
        self._vprint(f"  CML contacts: {len(contacts_cml)}, pairs: {len(pairs_cml) if pairs_cml is not None else 0}")

        comparator = DataFrameComparator(
            tolerant_numeric=True,
            rtol=self.rtol_coords,
            atol=self.atol_coords,
            sort_keys=["label"],
        )

        tag = self.session_tag
        acq = self.acq_label
        suffix = "contacts" if self.acquisition == "contacts" else "pairs"

        # Enumerate every space BIDS has for this session.
        spaces = self.reader.list_available_spaces()
        if not spaces:
            fallback = self.reader.space
            spaces = [fallback] if fallback else []
        self._vprint(f"  BIDS spaces available: {spaces}")

        # For pairs we need bipolar channels once; skip the whole run if missing.
        ch_bip = None
        if self.acquisition == "pairs":
            self._vprint(f"  Loading BIDS bipolar channels...")
            ch_bip = load_bids_channels(self.reader, acquisition="bipolar")
            ch_bip = ch_bip[ch_bip["name"].astype(str).str.contains("-")]
            self._vprint(f"  Bipolar channels found: {len(ch_bip)}")

        per_summary: List[pd.DataFrame] = []
        per_detail: List[pd.DataFrame] = []
        per_mismatches: List[pd.DataFrame] = []
        montage_rows: List[Dict[str, Any]] = []
        last_res = None

        for bids_space in spaces:
            cml_key = BIDS_TO_CML_SPACE.get(bids_space)
            if cml_key is None:
                self._vprint(f"  WARNING: unknown BIDS space '{bids_space}' — skipping")
                montage_rows.append({
                    "subject": self.subject, "experiment": self.experiment,
                    "session": self.session, "acquisition": acq,
                    "space": bids_space, "skipped": True, "reason": "unknown_space",
                })
                continue

            self._vprint(f"\n  --- {suffix} comparison for space={bids_space} (cml_key={cml_key}) ---")
            try:
                elec = self.reader.load_electrodes(space=bids_space)
            except Exception as e:
                self._vprint(f"  WARNING: load_electrodes(space={bids_space}) failed: {e}")
                montage_rows.append({
                    "subject": self.subject, "experiment": self.experiment,
                    "session": self.session, "acquisition": acq,
                    "space": bids_space, "skipped": True, "reason": f"load_failed: {e}",
                })
                continue

            if self.acquisition == "contacts":
                df_bids = prep_contacts(elec, cml_key=cml_key)
                res = comparator.compare(
                    contacts_cml, df_bids,
                    label_a="CML", label_b="BIDS", space=bids_space,
                    subject=self.subject, experiment=self.experiment,
                    session=self.session, return_aligned=True,
                )
            else:  # pairs
                if ch_bip is None or ch_bip.empty:
                    self._vprint(f"  Skipped: no bipolar channels")
                    montage_rows.append({
                        "subject": self.subject, "experiment": self.experiment,
                        "session": self.session, "acquisition": acq,
                        "space": bids_space, "skipped": True, "reason": "no_bipolar_channels",
                    })
                    continue
                if pairs_cml is None or pairs_cml.empty:
                    self._vprint(f"  Skipped: no CML pairs data")
                    montage_rows.append({
                        "subject": self.subject, "experiment": self.experiment,
                        "session": self.session, "acquisition": acq,
                        "space": bids_space, "skipped": True, "reason": "no_cml_pairs",
                    })
                    continue
                df_bids = prep_pairs(elec, ch_bip, cml_key=cml_key)
                res = comparator.compare(
                    pairs_cml, df_bids,
                    label_a="CML", label_b="BIDS", space=bids_space,
                    subject=self.subject, experiment=self.experiment,
                    session=self.session, return_aligned=True,
                )

            self._vprint(f"  space={bids_space} ok={res.ok}")
            per_summary.append(res.df_summary)
            per_detail.append(res.df_detail)
            per_mismatches.append(res.df_mismatches)
            montage_rows.append({
                "subject": self.subject, "experiment": self.experiment,
                "session": self.session, "acquisition": acq,
                "space": bids_space, "skipped": False, "reason": "",
            })
            last_res = res

        # Write concatenated per-session outputs (one row per space).
        if per_summary:
            self._save_df(pd.concat(per_summary, ignore_index=True),
                          f"df_{suffix}_summary_{tag}.csv")
            self._save_df(pd.concat(per_detail, ignore_index=True),
                          f"df_{suffix}_column_summary_{tag}.csv")
            self._save_df(pd.concat(per_mismatches, ignore_index=True),
                          f"df_{suffix}_mismatches_{tag}.csv")

        pd.DataFrame(montage_rows).to_csv(
            self._make_path(f"df_montage_status_{acq}"), index=False,
        )

        return last_res if last_res is not None else {"skipped": True, "reason": "no_spaces"}
