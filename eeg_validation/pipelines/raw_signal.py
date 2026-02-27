"""Raw (continuous) EEG signal comparison pipeline."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base import BasePipeline
from ..loaders.cml import load_cml_eeg_raw, load_cml_contacts_and_pairs
from ..loaders.bids import load_bids_raw, raw_to_xarray
from ..comparators.signal import SignalComparator


class RawSignalPipeline(BasePipeline):
    """Compare CML vs BIDS continuous (raw) EEG for one session."""

    def _output_paths(self) -> List[str]:
        if self.is_intracranial:
            paths = []
            for acq in ("mono", "bi"):
                paths.append(self._make_path(f"df_raw_{acq}"))
                paths.append(self._make_path(f"df_raw_summary_{acq}"))
                paths.append(self._make_path(f"df_time_{acq}"))
            return paths
        return [
            self._make_path("df_raw"),
            self._make_path("df_raw_summary"),
            self._make_path("df_raw_time"),
        ]

    def _run(self) -> Dict[str, Any]:
        comparator = SignalComparator()
        results = {}

        if self.is_intracranial:
            contacts, pairs = load_cml_contacts_and_pairs(
                self.subject, self.experiment, self.session,
                self.localization, self.montage,
            )

            for acq_tag, scheme, bids_acq in [
                ("mono", contacts, "monopolar"),
                ("bi", pairs, "bipolar"),
            ]:
                # CML raw
                eeg_cml = load_cml_eeg_raw(
                    self.subject, self.experiment, self.session,
                    localization=self.localization, montage=self.montage,
                    scheme=scheme,
                )

                # BIDS raw via BIDSReader
                raw_bids = load_bids_raw(self.reader, acquisition=bids_acq)
                eeg_bids = raw_to_xarray(raw_bids, time_unit_ms=True)

                result = comparator.compare(
                    eeg_bids, eeg_cml,
                    label_a="BIDS", label_b="CMLReader",
                    subject=self.subject, experiment=self.experiment,
                    session=self.session,
                )

                self._save_df(result.extras["df_raw"], f"df_raw_{self.session_tag}_{acq_tag}.csv")
                self._save_df(result.extras["df_raw_summary"], f"df_raw_summary_{self.session_tag}_{acq_tag}.csv")
                self._save_df(result.extras["df_time"], f"df_time_{self.session_tag}_{acq_tag}.csv")
                results[acq_tag] = result
        else:
            # CML raw (scalp)
            eeg_cml = load_cml_eeg_raw(
                self.subject, self.experiment, self.session,
            )

            # BIDS raw via BIDSReader (scalp — no acquisition)
            raw_bids = load_bids_raw(self.reader)
            eeg_bids = raw_to_xarray(raw_bids, time_unit_ms=True)

            result = comparator.compare(
                eeg_bids, eeg_cml,
                label_a="BIDS", label_b="CMLReader",
                subject=self.subject, experiment=self.experiment,
                session=self.session,
            )

            self._save_df(result.extras["df_raw"], f"df_raw_{self.session_tag}.csv")
            self._save_df(result.extras["df_raw_summary"], f"df_raw_summary_{self.session_tag}.csv")
            self._save_df(result.extras["df_time"], f"df_raw_time_{self.session_tag}.csv")
            results["eeg"] = result

        return results
