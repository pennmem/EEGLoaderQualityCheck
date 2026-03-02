"""Montage (contacts + pairs) comparison pipeline.

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
    """Compare CML vs BIDS contacts and pairs for one iEEG session."""

    def __init__(self, *args, atol_coords: float = 1e-3, rtol_coords: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.atol_coords = atol_coords
        self.rtol_coords = rtol_coords

    def _output_paths(self) -> List[str]:
        return [self._make_path("df_montage_summary")]

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

        results = {}
        elec = None
        elec_space = None

        # --- Contacts ---
        self._vprint(f"\n  --- Contacts comparison ---")
        try:
            self._vprint(f"  Loading BIDS electrodes...")
            elec = load_bids_electrodes(self.reader)
            elec_space = self.reader.space or "unknown"
            self._vprint(f"  BIDS electrodes loaded: {len(elec)} contacts, space='{elec_space}'")

            contact_bids = prep_contacts(elec, elec_space=elec_space)
            self._vprint(f"  Prepped BIDS contacts: {len(contact_bids)} rows")

            self._vprint(f"  Comparing CML vs BIDS contacts...")
            res = comparator.compare(
                contacts_cml, contact_bids,
                label_a="CML", label_b="BIDS",
                subject=self.subject, experiment=self.experiment,
                session=self.session, return_aligned=True,
            )
            self._vprint(f"  Contacts comparison complete (match={res.match})")

            tag = self.session_tag
            self._save_df(res.df_summary, f"df_contacts_summary_{tag}.csv")
            self._save_df(res.df_detail, f"df_contacts_column_summary_{tag}.csv")
            self._save_df(res.df_mismatches, f"df_contacts_mismatches_{tag}.csv")
            results["contacts"] = res

        except FileNotFoundError:
            self._vprint(f"  Skipped contacts: no electrodes file found")
            results["contacts"] = {"skipped": True, "reason": "no_electrodes"}
        except Exception as e:
            self._vprint(f"  Contacts comparison failed: {e}")
            results["contacts"] = {"skipped": True, "reason": "error", "error": str(e)}

        # --- Pairs ---
        self._vprint(f"\n  --- Pairs comparison ---")
        try:
            if elec is None:
                self._vprint(f"  Loading BIDS electrodes (not loaded yet)...")
                elec = load_bids_electrodes(self.reader)
                elec_space = self.reader.space or "unknown"

            self._vprint(f"  Loading BIDS bipolar channels...")
            ch_bip = load_bids_channels(self.reader, acquisition="bipolar")
            ch_bip = ch_bip[ch_bip["name"].astype(str).str.contains("-")]
            self._vprint(f"  Bipolar channels found: {len(ch_bip)}")

            if ch_bip.empty:
                self._vprint(f"  Skipped pairs: no bipolar channels")
                results["pairs"] = {"skipped": True, "reason": "no_bipolar_channels"}
            elif pairs_cml is None or pairs_cml.empty:
                self._vprint(f"  Skipped pairs: no CML pairs data")
                results["pairs"] = {"skipped": True, "reason": "no_cml_pairs"}
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
                self._vprint(f"  Pairs comparison complete (match={res.match})")

                tag = self.session_tag
                self._save_df(res.df_summary, f"df_pairs_summary_{tag}.csv")
                self._save_df(res.df_detail, f"df_pairs_column_summary_{tag}.csv")
                self._save_df(res.df_mismatches, f"df_pairs_mismatches_{tag}.csv")
                results["pairs"] = res

        except FileNotFoundError:
            self._vprint(f"  Skipped pairs: no electrodes file found")
            results["pairs"] = {"skipped": True, "reason": "no_electrodes"}
        except Exception as e:
            self._vprint(f"  Pairs comparison failed: {e}")
            results["pairs"] = {"skipped": True, "reason": "error", "error": str(e)}

        # --- Summary marker ---
        pd.DataFrame([{
            "subject": self.subject,
            "experiment": self.experiment,
            "session": self.session,
            "electrodes_space_used": elec_space,
            "contacts_skipped": isinstance(results.get("contacts"), dict) and results["contacts"].get("skipped", False),
            "pairs_skipped": isinstance(results.get("pairs"), dict) and results["pairs"].get("skipped", False),
        }]).to_csv(self._make_path("df_montage_summary"), index=False)

        return results