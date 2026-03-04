"""Raw (continuous) EEG signal comparison pipeline."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from .base import BasePipeline
from ..loaders.cml import load_cml_eeg_raw, load_cml_contacts_and_pairs
from ..loaders.bids import load_bids_raw, raw_to_xarray, convert_unit
from ..comparators.signal import SignalComparator
import numpy as np


class RawSignalPipeline(BasePipeline):
    """Compare CML vs BIDS continuous (raw) EEG for one session.

    For iEEG, ``acquisition`` must be ``"contacts"`` or ``"pairs"``.
    For scalp EEG, leave ``acquisition=None``.
    """

    def _output_paths(self) -> List[str]:
        tag = self.session_tag
        acq = self.acq_label
        return [
            os.path.join(self.out_path, f"df_raw_{tag}_{acq}.csv"),
            os.path.join(self.out_path, f"df_raw_summary_{tag}_{acq}.csv"),
            os.path.join(self.out_path, f"df_time_{tag}_{acq}.csv"),
        ]

    def _run(self) -> Dict[str, Any]:
        comparator = SignalComparator()

        scheme = None
        if self.is_intracranial:
            self._vprint(f"  Loading CML contacts and pairs...")
            contacts, pairs = load_cml_contacts_and_pairs(
                self.subject, self.experiment, self.session,
                self.localization, self.montage,
            )
            scheme = contacts if self.acquisition == "contacts" else pairs
            self._vprint(f"  Using scheme: {self.acquisition} ({len(scheme)} channels)")

        # CML raw
        self._vprint(f"  Loading CML raw EEG...")
        eeg_cml = load_cml_eeg_raw(
            self.subject, self.experiment, self.session,
            localization=self.localization, montage=self.montage,
            scheme=scheme,
        )
        self._vprint(f"  CML raw EEG shape: {eeg_cml.shape}")

        # BIDS raw
        self._vprint(f"  Loading BIDS raw EEG (acquisition={self.bids_acquisition})...")
        raw_bids = load_bids_raw(self.reader, acquisition=self.bids_acquisition)
        raw_bids.load_data()
        raw_bids._data *= 1e6
        raw_bids._data = np.round(raw_bids._data)
        self._vprint(f"After rounds")
        eeg_bids = raw_to_xarray(raw_bids, convert_ms=True)
        self._vprint(f"  BIDS raw EEG shape: {eeg_bids.shape}")

        # Compare
        self._vprint(f"  Comparing BIDS vs CML...")
        result = comparator.compare(
            eeg_bids, eeg_cml,
            label_a="BIDS", label_b="CMLReader",
            subject=self.subject, experiment=self.experiment,
            session=self.session,
        )
        self._vprint(f"  Comparison complete (ok={result.ok})")

        tag = self.session_tag
        acq = self.acq_label
        self._save_df(result.extras["df_raw"], f"df_raw_{tag}_{acq}.csv")
        self._save_df(result.extras["df_raw_summary"], f"df_raw_summary_{tag}_{acq}.csv")
        self._save_df(result.extras["df_time"], f"df_time_{tag}_{acq}.csv")

        return result
