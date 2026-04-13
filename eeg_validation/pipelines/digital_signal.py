"""Digital (integer) EEG comparison pipeline.

Compares the raw digitized samples from CMLReader against a BIDS EDF/BDF,
with no gain/offset/unit conversion. The CML side is loaded via
``cmlreaders.CMLReader.load_eeg()`` which returns the native integer
samples regardless of the underlying recording format (EDF, split-channel,
NSx, etc.). The BIDS side is read directly from the on-disk EDF or BDF.

The point is to verify the BIDS re-encoding round-tripped the original
integers losslessly (or at worst to within 1 LSB of unavoidable rescaling
rounding).

Saves a single summary CSV per acquisition. No plotting — plotting lives
in the user's notebook via ``eeg_validation.plotting.plot_comparison_results``,
which the digital summary's column names (``mean_abs_diff``, ``std_diff``,
``frac_diff_gt_1``, ...) are designed to feed directly.
"""

from __future__ import annotations

import glob
import os
from typing import Any, Dict, List, Optional

import cmlreaders as cml
import numpy as np
import pandas as pd

from .base import BasePipeline
from ..comparators.digital_signal import DigitalSignalComparator
from ..loaders.cml import load_cml_contacts_and_pairs


class DigitalSignalPipeline(BasePipeline):
    """Compare CMLReader digital integers vs BIDS EDF/BDF integers, per acquisition."""

    def _acqs_to_run(self) -> List[str]:
        """Which acquisitions to process. Respects ``self.acquisition``
        if set (``"contacts"``→monopolar, ``"pairs"``→bipolar),
        otherwise runs all applicable acquisitions."""
        if not self.is_intracranial:
            return ["eeg"]
        _ACQ_MAP = {"contacts": "monopolar", "pairs": "bipolar"}
        if self.acquisition is not None:
            return [_ACQ_MAP.get(self.acquisition, self.acquisition)]
        return ["monopolar", "bipolar"]

    def _output_paths(self) -> List[str]:
        acqs = self._acqs_to_run()
        tag = self.session_tag
        paths = [
            os.path.join(self.out_path, f"df_digital_summary_{tag}_{acq}.csv")
            for acq in acqs
        ]
        paths.append(os.path.join(self.out_path, f"df_digital_status_{tag}.csv"))
        return paths

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------
    def _load_cml_digital(self, acquisition: str):
        """Load CML integer samples via CMLReader.

        Returns ``("EDF", labels, [int64_array_per_channel])`` — the
        format tag is ``"EDF"`` because CMLReader's native integers live
        in the int16 (EDF) range for system 1/2 recordings.
        """
        reader = cml.CMLReader(
            subject=self.subject,
            experiment=self.experiment,
            session=int(self.session),
            localization=self.localization,
            montage=self.montage,
        )
        scheme = None
        if self.is_intracranial:
            contacts, pairs = load_cml_contacts_and_pairs(
                self.subject, self.experiment, self.session,
                self.localization, self.montage,
            )
            scheme = contacts if acquisition in ("monopolar", "contacts") else pairs

        eeg = reader.load_eeg(scheme=scheme)
        arr = np.asarray(eeg.data)
        if arr.ndim == 3:
            arr = np.squeeze(arr, axis=0)
        # CMLReader returns float64 but the values are integer LSBs.
        labels = list(eeg.channels)
        signals = [arr[i].astype(np.int64) for i in range(arr.shape[0])]
        return "EDF", labels, signals

    def _bids_edf_bdf_path(self, acquisition: str) -> Optional[str]:
        """Resolve the BIDS EDF or BDF for the given acquisition."""
        ses_dir = os.path.join(
            self.bids_root,
            f"sub-{self.subject}",
            f"ses-{self.session}",
            "ieeg" if self.is_intracranial else "eeg",
        )
        if not os.path.isdir(ses_dir):
            return None
        prefix = f"sub-{self.subject}_ses-{self.session}_task-{self.experiment}"
        if self.is_intracranial:
            prefix += f"_acq-{acquisition}_ieeg"
        else:
            prefix += "_eeg"
        for ext in (".bdf", ".edf"):
            pattern = os.path.join(ses_dir, prefix + ext)
            matches = glob.glob(pattern)
            if matches:
                return matches[0]
        return None

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    def _run(self) -> Dict[str, Any]:
        comparator = DigitalSignalComparator()

        acqs = self._acqs_to_run()
        status: List[tuple] = []
        results: Dict[str, Any] = {}
        tag = self.session_tag

        for acq in acqs:
            self._vprint(f"\n  === Acquisition: {acq} ===")

            # --- BIDS side ---
            bids_path = self._bids_edf_bdf_path(acq)
            self._vprint(f"    BIDS file: {bids_path}")
            if bids_path is None:
                self._vprint("    Skipped: no BIDS EDF/BDF found")
                status.append((acq, "skip", "no_bids_file"))
                self._save_df(pd.DataFrame(), f"df_digital_summary_{tag}_{acq}.csv")
                continue

            # --- CML side (via CMLReader) ---
            try:
                cml_source = self._load_cml_digital(acq)
                self._vprint(f"    CML loaded: {len(cml_source[1])} channels")
            except Exception as e:
                self._vprint(f"    Skipped: CMLReader load failed: {e}")
                status.append((acq, "skip", f"cml_load_error: {e}"))
                self._save_df(pd.DataFrame(), f"df_digital_summary_{tag}_{acq}.csv")
                continue

            # --- Compare ---
            try:
                result = comparator.compare(
                    cml_source,
                    bids_path,
                    label_a="CMLReader",
                    label_b="BIDS",
                    subject=self.subject,
                    experiment=self.experiment,
                    session=self.session,
                    acquisition=acq,
                )
                df_summary = result.df_summary
                self._save_df(df_summary, f"df_digital_summary_{tag}_{acq}.csv")
                results[acq] = df_summary
                status.append((acq, "ok" if result.ok else "diff", ""))
                self._vprint(f"    Done (ok={result.ok})")
            except Exception as e:
                self._vprint(f"    FAILED: {repr(e)}")
                status.append((acq, "fail", repr(e)))
                self._save_df(pd.DataFrame(), f"df_digital_summary_{tag}_{acq}.csv")

        df_status = pd.DataFrame(status, columns=["acquisition", "status", "detail"])
        self._save_df(df_status, f"df_digital_status_{tag}.csv")
        results["status"] = df_status
        return results
