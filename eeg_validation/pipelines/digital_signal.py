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
from ..comparators.digital_signal import (
    DigitalSignalComparator,
    _FORMAT_DIGITAL_RANGE,
    _read_edf_bdf_digital,
    read_edf_bdf_units,
)
from ..digital_units import compute_gain_offset, dim_to_si_scale
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
    def _load_cml_digital(self, acquisition: str, bids_path: Optional[str] = None):
        """Load CML integer samples for comparison against the BIDS EDF/BDF.

        Two source regimes need different handling:

        - **iEEG (System 1/2):** ``CMLReader.load_eeg()`` returns the native
          EDF integer LSBs as float64. We cast to int and tag the format from
          the sample range — 16-bit data stays within ±32768 → ``"EDF"``;
          genuinely 24-bit data → ``"BDF"`` (matching the BIDS container as
          ``(1, 1)``, no spurious rescale).

        - **Scalp, EGI ``.raw``/``.mff`` source:** ``load_eeg()`` returns
          **physical Volts**, not integer LSBs (the EGI recording has no EDF
          integers). A naive ``astype(int64)`` would floor every sample to 0.
          Instead we *reconstruct* the digital ints by requantizing the CML
          Volts with the **same per-channel gain/offset the BIDS BDF was
          written with**, recovered from the BDF header
          (``digital = round((µV - offset) / gain)``). The result is tagged
          ``"BDF"`` so it compares against the BIDS BDF as ``(1, 1)``; any
          residual diff is just requantization rounding (≤1 LSB).

        - **Scalp, native ``.bdf``/``.edf`` source:** the recording already
          *is* integer LSBs, and the BIDS file is a bit-exact copy of it
          (``ScalpBIDSConverter._write_eeg_from_bdf``). Reconstructing from
          MNE Volts would inject avoidable rounding, so we skip ``load_eeg()``
          entirely and read the source EDF/BDF's integer samples directly.
        """
        reader = cml.CMLReader(
            subject=self.subject,
            experiment=self.experiment,
            session=int(self.session),
            localization=self.localization,
            montage=self.montage,
        )
        if not self.is_intracranial:
            # Scalp: only EGI .raw/.mff need volts→ints reconstruction;
            # native .bdf/.edf sources are already digital, so read them
            # straight from disk to avoid requantization rounding.
            data_format, source_path = self._cml_scalp_source(reader)
            if data_format not in (".raw", ".mff"):
                return _read_edf_bdf_digital(source_path)
            eeg = reader.load_eeg()
            arr = np.asarray(eeg.data)
            if arr.ndim == 3:
                arr = np.squeeze(arr, axis=0)
            labels = list(eeg.channels)
            return self._reconstruct_scalp_digital(arr, labels, bids_path)

        contacts, pairs = load_cml_contacts_and_pairs(
            self.subject, self.experiment, self.session,
            self.localization, self.montage,
        )
        scheme = contacts if acquisition in ("monopolar", "contacts") else pairs

        eeg = reader.load_eeg(scheme=scheme)
        arr = np.asarray(eeg.data)
        if arr.ndim == 3:
            arr = np.squeeze(arr, axis=0)
        labels = list(eeg.channels)

        # iEEG: native integer LSBs returned as float64.
        signals = [arr[i].astype(np.int64) for i in range(arr.shape[0])]
        # Detect the native digital format from the sample range: a 16-bit
        # recording cannot exceed the int16 range, a 24-bit one will.
        peak = max((int(np.abs(s).max()) for s in signals if s.size), default=0)
        fmt = "BDF" if peak > _FORMAT_DIGITAL_RANGE["EDF"] else "EDF"
        return fmt, labels, signals

    def _cml_scalp_source(self, reader):
        """Return ``(data_format, source_path)`` for the scalp recording.

        ``data_format`` is the source extension as recorded in ``sources.json``
        (e.g. ``".bdf"``, ``".raw"``, ``".mff"``). ``source_path`` is the
        on-disk recording in the sibling ``noreref/`` directory — the same file
        ``CMLReader`` hands to MNE. Used to decide whether the digital samples
        can be read directly (native EDF/BDF) or must be reconstructed (EGI).
        """
        from cmlreaders.path_finder import PathFinder

        finder = PathFinder(
            subject=self.subject,
            experiment=self.experiment,
            session=int(self.session),
        )
        sources_json = finder.find("sources")
        df = pd.read_json(sources_json, orient="index")
        data_format = str(df["data_format"].iloc[0])
        basename = str(df.index[0])
        source_path = os.path.join(
            os.path.dirname(sources_json), "noreref", basename,
        )
        return data_format, source_path

    def _reconstruct_scalp_digital(self, arr, labels, bids_path):
        """Requantize CML Volts into the BIDS BDF digital domain.

        Uses the target BDF header's per-channel ``physical/digital`` range to
        recover ``gain, offset`` (``physical = digital*gain + offset``) and
        inverts: ``digital = round((cml_µV - offset) / gain)``. Channels absent
        from the BDF header are dropped (the comparator only uses common ones).
        """
        if bids_path is None:
            raise ValueError("scalp digital reconstruction requires bids_path")
        units = read_edf_bdf_units(bids_path)
        out_labels = []
        signals = []
        for i, lbl in enumerate(labels):
            if lbl not in units:
                continue
            pmin, pmax, dmin, dmax, dim = units[lbl]
            gain, offset = compute_gain_offset(pmin, pmax, dmin, dmax)
            # CML Volts → physical units of the BDF header (e.g. µV).
            cml_phys = arr[i].astype(np.float64) / dim_to_si_scale(dim)
            ints = np.round((cml_phys - offset) / gain).astype(np.int64)
            out_labels.append(lbl)
            signals.append(ints)
        return "BDF", out_labels, signals

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
        task = self.experiment if self.is_intracranial else self.experiment.lower()
        prefix = f"sub-{self.subject}_ses-{self.session}_task-{task}"
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
                cml_source = self._load_cml_digital(acq, bids_path)
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
