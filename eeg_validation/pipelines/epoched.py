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

    Loads ALL epochs once per acquisition stream, then slices per event
    type for comparison.  This avoids re-reading the raw BDF file for
    every event type.

    For each event type:
      1. CML: filter events → dedupe by eegoffset → slice from bulk array
      2. BIDS: filter events → dedupe by sample  → slice from bulk array
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
            bids_acq = acq_tag if self.is_intracranial else None
            scheme = cml_schemes.get(acq_tag)

            # ==============================================================
            # Prepare per-type events (dedupe within each type)
            # ==============================================================
            cml_events_by_type = {}
            bids_events_by_type = {}

            for etype in types_to_run:
                evs_cml_t = evs_cml[evs_cml["type"] == etype].copy()
                if not evs_cml_t.empty:
                    cml_events_by_type[etype] = dedupe_events_by_sample(evs_cml_t, "eegoffset")

                evs_bids_t, _ = filter_events_df(evs_bids, etype)
                if not evs_bids_t.empty:
                    bids_events_by_type[etype] = evs_bids_t.drop_duplicates(
                        subset=["sample"], keep="first",
                    )

            self._vprint(
                f"  Event types with CML data: {len(cml_events_by_type)}, "
                f"with BIDS data: {len(bids_events_by_type)}"
            )

            # ==============================================================
            # Bulk-load CML epochs (one read of the raw file)
            # ==============================================================
            eeg_cml_all = None
            evs_cml_combined = pd.DataFrame()

            if cml_events_by_type:
                evs_cml_combined = pd.concat(
                    [cml_events_by_type[t] for t in types_to_run if t in cml_events_by_type],
                    ignore_index=True,
                )
                self._vprint(f"  Loading ALL CML epochs ({len(evs_cml_combined)} events)...")
                eeg_cml_all = load_cml_eeg_epoched(
                    self.subject, self.experiment, self.session,
                    evs_cml_combined, self.tmin, self.tmax,
                    localization=self.localization, montage=self.montage,
                    scheme=scheme,
                )
                self._vprint(f"  CML bulk EEG shape: {eeg_cml_all.shape}")

            # ==============================================================
            # Bulk-load BIDS epochs (one read of the raw file)
            # ==============================================================
            eeg_bids_all = None
            evs_bids_combined = pd.DataFrame()

            if bids_events_by_type:
                evs_bids_combined = pd.concat(
                    [bids_events_by_type[t] for t in types_to_run if t in bids_events_by_type],
                    ignore_index=True,
                ).sort_values("sample").reset_index(drop=True)

                self._vprint(f"  Loading ALL BIDS epochs ({len(evs_bids_combined)} events)...")
                epochs_bids_all = self.reader.load_epochs(
                    tmin=self.tmin / 1000.0,
                    tmax=self.tmax / 1000.0,
                    events=evs_bids_combined,
                    acquisition=bids_acq,
                    baseline=None,
                    preload=True,
                )
                self._vprint(f"  BIDS bulk epochs: {len(epochs_bids_all)} epochs, {len(epochs_bids_all.ch_names)} channels")

                # Pick only EEG/iEEG channels (once)
                if self.is_intracranial:
                    picks = mne.pick_types(epochs_bids_all.info, ieeg=True, eeg=False, eog=False, misc=False)
                else:
                    picks = mne.pick_types(epochs_bids_all.info, eeg=True, eog=False, misc=False)
                if len(picks) == 0:
                    picks = np.arange(len(epochs_bids_all.ch_names))
                epochs_bids_all = epochs_bids_all.pick(picks)
                self._vprint(f"  After channel pick: {len(epochs_bids_all.ch_names)} channels")

                # Convert to PTSA once
                eeg_bids_all = epochs_to_ptsa(epochs_bids_all, evs_bids_combined)
                eeg_bids_all = eeg_bids_all.assign_coords(time=eeg_bids_all["time"] * 1000.0)
                eeg_bids_all["time"].attrs["units"] = "ms"
                self._vprint(f"  BIDS bulk EEG shape: {eeg_bids_all.shape}")

                del epochs_bids_all

            # ==============================================================
            # Per event type: slice and compare
            # ==============================================================
            all_raw, all_summary, all_time, status = [], [], [], []

            for etype in types_to_run:
                try:
                    self._vprint(f"    Processing event type: {etype}")

                    if etype not in cml_events_by_type:
                        self._vprint(f"      Skipped: no CML events for type '{etype}'")
                        status.append((acq_tag, etype, "skip", "no_cml_events"))
                        continue
                    if etype not in bids_events_by_type:
                        self._vprint(f"      Skipped: no BIDS events for type '{etype}'")
                        status.append((acq_tag, etype, "skip", "no_bids_events"))
                        continue

                    # CML slice
                    cml_mask = evs_cml_combined["type"].values == etype
                    eeg_cml_t = eeg_cml_all.isel(event=np.where(cml_mask)[0])
                    self._vprint(f"      CML slice: {eeg_cml_t.shape}")

                    # BIDS slice
                    bids_mask = evs_bids_combined["trial_type"].values == etype
                    eeg_bids_t = eeg_bids_all.isel(event=np.where(bids_mask)[0])
                    self._vprint(f"      BIDS slice: {eeg_bids_t.shape}")

                    # Compare
                    self._vprint(f"      Comparing BIDS vs CML...")
                    result = comparator.compare(
                        eeg_bids_t, eeg_cml_t,
                        label_a="BIDS", label_b="CMLReader",
                        subject=self.subject, experiment=self.experiment,
                        session=self.session,
                    )
                    self._vprint(f"      Comparison complete (ok={result.ok})")

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

            # Free bulk arrays before saving
            del eeg_cml_all, eeg_bids_all
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
