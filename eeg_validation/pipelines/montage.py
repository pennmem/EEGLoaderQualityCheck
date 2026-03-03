"""Montage (contacts or pairs) comparison pipeline.

Uses BIDSReader.load_electrodes() and BIDSReader.load_channels() instead
of manually constructing BIDSPaths.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from .base import BasePipeline
from ..loaders.cml import load_cml_contacts_and_pairs
from ..loaders.bids import load_bids_electrodes, load_bids_channels
from ..preparers.montage import prep_contacts, prep_pairs
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
        return [self._make_path(f"df_montage_summary_{acq}")]

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

        # Load electrodes (needed for both contacts and pairs)
        self._vprint(f"  Loading BIDS electrodes...")
        elec = load_bids_electrodes(self.reader)
        elec_space = self.reader.space or "unknown"
        self._vprint(f"  BIDS electrodes loaded: {len(elec)} contacts, space='{elec_space}'")

        result = {}

        if self.acquisition == "contacts":
            self._vprint(f"\n  --- Contacts comparison ---")
            contact_bids = prep_contacts(elec, elec_space=elec_space)
            self._vprint(f"  Prepped BIDS contacts: {len(contact_bids)} rows")

            self._vprint(f"  Comparing CML vs BIDS contacts...")
            res = comparator.compare(
                contacts_cml, contact_bids,
                label_a="CML", label_b="BIDS",
                subject=self.subject, experiment=self.experiment,
                session=self.session, return_aligned=True,
            )
            self._vprint(f"  Contacts comparison complete (ok={res.ok})")

            self._save_df(res.df_summary, f"df_contacts_summary_{tag}.csv")
            self._save_df(res.df_detail, f"df_contacts_column_summary_{tag}.csv")
            self._save_df(res.df_mismatches, f"df_contacts_mismatches_{tag}.csv")
            result = res

        elif self.acquisition == "pairs":
            self._vprint(f"\n  --- Pairs comparison ---")
            self._vprint(f"  Loading BIDS bipolar channels...")
            ch_bip = load_bids_channels(self.reader, acquisition="bipolar")
            ch_bip = ch_bip[ch_bip["name"].astype(str).str.contains("-")]
            self._vprint(f"  Bipolar channels found: {len(ch_bip)}")

            if ch_bip.empty:
                self._vprint(f"  Skipped: no bipolar channels")
                result = {"skipped": True, "reason": "no_bipolar_channels"}
            elif pairs_cml is None or pairs_cml.empty:
                self._vprint(f"  Skipped: no CML pairs data")
                result = {"skipped": True, "reason": "no_cml_pairs"}
            else:
                pairs_bids = prep_pairs(elec, ch_bip, elec_space=elec_space)
                self._vprint(f"  Prepped BIDS pairs: {len(pairs_bids)} rows")

                self._vprint(f"  Comparing CML vs BIDS pairs...")
                res = comparator.compare(
                    pairs_cml, pairs_bids,
                    label_a="CML", label_b="BIDS",
                    subject=self.subject, experiment=self.experiment,
                    session=self.session, return_aligned=True,
                )
                self._vprint(f"  Pairs comparison complete (ok={res.ok})")

                self._save_df(res.df_summary, f"df_pairs_summary_{tag}.csv")
                self._save_df(res.df_detail, f"df_pairs_column_summary_{tag}.csv")
                self._save_df(res.df_mismatches, f"df_pairs_mismatches_{tag}.csv")
                result = res

        # Summary marker
        skipped = isinstance(result, dict) and result.get("skipped", False)
        pd.DataFrame([{
            "subject": self.subject,
            "experiment": self.experiment,
            "session": self.session,
            "acquisition": acq,
            "electrodes_space_used": elec_space,
            "skipped": skipped,
        }]).to_csv(self._make_path(f"df_montage_summary_{acq}"), index=False)

        return result
