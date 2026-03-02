"""Epoched EEG signal comparison pipeline.

Uses BIDSReader.load_epochs(events=filtered_df) to epoch BIDS data,
mirroring how CMLReader.load_eeg(events=) works.
"""

from __future__ import annotations

import gc
from typing import Any, Dict, List, Optional, Sequence, Union

import mne
import numpy as np
import pandas as pd

from .base import BasePipeline
from ..loaders.cml import load_cml_events, load_cml_eeg_epoched, load_cml_contacts_and_pairs
from ..loaders.bids import (
    load_bids_events,
    load_bids_epochs,
    load_bids_raw,
    epochs_to_ptsa,
    filter_events_df,
)
from ..preparers.events import dedupe_events_by_sample
from ..comparators.signal import SignalComparator

from bidsreader import BIDSReader


class EpochedPipeline(BasePipeline):
    """Compare CML vs BIDS epoched EEG for one session, per event type.

    For each event type:
      1. CML: filter events → dedupe by eegoffset → CMLReader.load_eeg(events=)
      2. BIDS: filter events → BIDSReader.load_epochs(events=filtered_df)
              → BIDSReader.mne_epochs_to_ptsa(epochs, events)
      3. Compare via SignalComparator
    """

    def __init__(
        self,
        *args,
        evs_types: Optional[Sequence[str]] = None,
        tmin: float = 0,
        tmax: float = 1000,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.evs_types = evs_types
        self.tmin = tmin
        self.tmax = tmax

    def _output_paths(self) -> List[str]:
        acqs = ["monopolar", "bipolar"] if self.is_intracranial else ["eeg"]
        paths = []
        for acq in acqs:
            paths.append(self._make_path(f"df_epoch_{acq}"))
            paths.append(self._make_path(f"df_epoch_summary_{acq}"))
            paths.append(self._make_path(f"df_epoch_time_{acq}"))
        return paths

    def _run(self) -> Dict[str, Any]:
        # Load CML events once
        self._vprint(f"  Loading CML events...")
        evs_cml = load_cml_events(
            self.subject, self.experiment, self.session,
            localization=self.localization, montage=self.montage,
        )
        self._vprint(f"  CML events loaded: {len(evs_cml)} rows")

        # Load BIDS events once (EEG-aligned events with sample/trial_type columns)
        self._vprint(f"  Loading BIDS events (event_type={self.reader.eeg_type})...")
        evs_bids = load_bids_events(self.reader, event_type=self.reader.eeg_type)
        self._vprint(f"  BIDS events loaded: {len(evs_bids)} rows, columns={list(evs_bids.columns)}")

        types_to_run = (
            sorted(set(self.evs_types))
            if self.evs_types
            else sorted(evs_cml["type"].dropna().unique())
        )
        self._vprint(f"  Event types to process: {types_to_run}")
        self._vprint(f"  Epoch window: tmin={self.tmin}, tmax={self.tmax}")

        # CML schemes for iEEG
        cml_schemes = {}
        if self.is_intracranial:
            self._vprint(f"  Loading CML contacts and pairs (intracranial)...")
            contacts, pairs = load_cml_contacts_and_pairs(
                self.subject, self.experiment, self.session,
                self.localization, self.montage,
            )
            cml_schemes = {"monopolar": contacts, "bipolar": pairs}
            self._vprint(f"  Contacts: {len(contacts)}, Pairs: {len(pairs)}")

        # Determine acquisition streams
        acq_tags = ["monopolar", "bipolar"] if self.is_intracranial else ["eeg"]
        self._vprint(f"  Acquisition streams: {acq_tags}")

        comparator = SignalComparator()
        results_all = {}

        for acq_tag in acq_tags:
            self._vprint(f"\n  --- Acquisition: {acq_tag} ---")
            # For BIDS: determine acquisition param for BIDSReader
            bids_acq = acq_tag if self.is_intracranial else None

            all_raw, all_summary, all_time, status = [], [], [], []

            for etype in types_to_run:
                try:
                    self._vprint(f"    Processing event type: {etype}")

                    # ---- CML: filter + dedupe + epoch ----
                    evs_cml_t = evs_cml[evs_cml["type"] == etype].copy()
                    if evs_cml_t.empty:
                        self._vprint(f"      Skipped: no CML events for type '{etype}'")
                        status.append((acq_tag, etype, "skip", "no_cml_events"))
                        continue

                    evs_cml_t = dedupe_events_by_sample(evs_cml_t, "eegoffset")
                    self._vprint(f"      CML events after dedupe: {len(evs_cml_t)}")
                    scheme = cml_schemes.get(acq_tag)
                    self._vprint(f"      Loading CML epoched EEG...")
                    eeg_cml = load_cml_eeg_epoched(
                        self.subject, self.experiment, self.session,
                        evs_cml_t, self.tmin, self.tmax,
                        localization=self.localization, montage=self.montage,
                        scheme=scheme,
                    )
                    self._vprint(f"      CML EEG shape: {eeg_cml.shape}")

                    # ---- BIDS: filter events DF → load_epochs(events=) ----
                    evs_bids_t, _ = filter_events_df(evs_bids, etype)
                    if evs_bids_t.empty:
                        self._vprint(f"      Skipped: no BIDS events for type '{etype}'")
                        status.append((acq_tag, etype, "skip", "no_bids_events"))
                        del eeg_cml
                        gc.collect()
                        continue

                    # Dedupe by sample for BIDS too
                    evs_bids_t = evs_bids_t.drop_duplicates(subset=["sample"], keep="first")
                    self._vprint(f"      BIDS events after dedupe: {len(evs_bids_t)}")

                    # BIDSReader.load_epochs accepts an events DataFrame
                    # with 'sample' and 'trial_type' columns
                    self._vprint(f"      Loading BIDS epochs...")
                    epochs_bids = self.reader.load_epochs(
                        tmin=self.tmin / 1000.0,
                        tmax=self.tmax / 1000.0,
                        events=evs_bids_t,
                        acquisition=bids_acq,
                        baseline=None,
                        preload=True,
                    )
                    self._vprint(f"      BIDS epochs: {len(epochs_bids)} epochs, {len(epochs_bids.ch_names)} channels")

                    # Pick only EEG/iEEG channels
                    if self.is_intracranial:
                        picks = mne.pick_types(epochs_bids.info, ieeg=True, eeg=False, eog=False, misc=False)
                    else:
                        picks = mne.pick_types(epochs_bids.info, eeg=True, eog=False, misc=False)
                    if len(picks) == 0:
                        picks = np.arange(len(epochs_bids.ch_names))
                    epochs_bids = epochs_bids.pick(picks)
                    self._vprint(f"      After channel pick: {len(epochs_bids.ch_names)} channels")

                    # Convert to PTSA TimeSeries via BIDSReader static method
                    eeg_bids = epochs_to_ptsa(epochs_bids, evs_bids_t)
                    eeg_bids = eeg_bids.assign_coords(time=eeg_bids["time"] * 1000.0)
                    eeg_bids["time"].attrs["units"] = "ms"
                    self._vprint(f"      BIDS EEG shape: {eeg_bids.shape}")

                    # ---- Compare ----
                    self._vprint(f"      Comparing BIDS vs CML...")
                    result = comparator.compare(
                        eeg_bids, eeg_cml,
                        label_a="BIDS", label_b="CMLReader",
                        subject=self.subject, experiment=self.experiment,
                        session=self.session,
                    )
                    self._vprint(f"      Comparison complete (match={result.ok})")

                    for key, container in [
                        ("df_raw", all_raw),
                        ("df_raw_summary", all_summary),
                        ("df_time", all_time),
                    ]:
                        df = result.extras.get(key)
                        if df is not None and not df.empty:
                            df = df.copy()
                            df["event_type"] = etype
                            df["acquisition"] = acq_tag
                            container.append(df)

                    status.append((acq_tag, etype, "ok", ""))

                except Exception as e:
                    self._vprint(f"      FAILED: {repr(e)}")
                    status.append((acq_tag, etype, "fail", repr(e)))
                finally:
                    gc.collect()

            # Save per acquisition
            self._vprint(f"\n  Saving results for {acq_tag}...")
            tag = self.session_tag
            df_raw = pd.concat(all_raw, ignore_index=True) if all_raw else pd.DataFrame()
            df_summary = pd.concat(all_summary, ignore_index=True) if all_summary else pd.DataFrame()
            df_time = pd.concat(all_time, ignore_index=True) if all_time else pd.DataFrame()
            df_status = pd.DataFrame(status, columns=["acquisition", "event_type", "status", "detail"])

            self._save_df(df_raw, f"df_epoch_{tag}_{acq_tag}.csv")
            self._save_df(df_summary, f"df_epoch_summary_{tag}_{acq_tag}.csv")
            self._save_df(df_time, f"df_epoch_time_{tag}_{acq_tag}.csv")
            self._save_df(df_status, f"df_epoch_status_{tag}_{acq_tag}.csv")

            results_all[acq_tag] = {
                "df_raw": df_raw,
                "df_raw_summary": df_summary,
                "df_time": df_time,
                "status": df_status,
            }

        return results_all
