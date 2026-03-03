"""Epoched EEG signal comparison pipeline.

Uses BIDSReader.load_epochs(events=filtered_df) to epoch BIDS data,
mirroring how CMLReader.load_eeg(events=) works.

Strategy: attempt bulk epoch load for all event types at once.
If it fails, probe each type individually to find the bad one(s),
remove them, and retry.  This gives the speed of a single file read
while gracefully handling types whose epochs extend past the recording.
"""

from __future__ import annotations

import gc
import os
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
    convert_unit
)
from ..preparers.events import dedupe_events_by_sample
from ..comparators.signal import SignalComparator

from bidsreader import BIDSReader


class EpochedPipeline(BasePipeline):
    """Compare CML vs BIDS epoched EEG for one session, per event type.

    Loads ALL epochs once, then slices per event type for comparison.
    If the bulk load fails, it identifies which event types caused the
    failure and retries without them.

    For iEEG, ``acquisition`` must be ``"contacts"`` or ``"pairs"``.
    For scalp EEG, leave ``acquisition=None``.
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
        tag = self.session_tag
        acq = self.acq_label
        paths = []
        for prefix in ("df_epoch", "df_epoch_summary", "df_epoch_time", "df_epoch_status"):
            paths.append(os.path.join(self.out_path, f"{prefix}_{tag}_{acq}.csv"))
        return paths

    # ==================================================================
    # Bulk load with retry helpers
    # ==================================================================

    def _bulk_load_cml(self, cml_events_by_type, types_to_run, scheme, status):
        """Attempt bulk CML epoch load; on failure, find and remove bad types."""
        acq = self.acq_label
        remaining = [t for t in types_to_run if t in cml_events_by_type]
        failed = set()

        while remaining:
            evs_combined = pd.concat(
                [cml_events_by_type[t] for t in remaining],
                ignore_index=True,
            ).drop_duplicates(subset=["eegoffset"], keep="first") \
             .sort_values("eegoffset").reset_index(drop=True)

            self._vprint(f"  Bulk CML load: {len(evs_combined)} events, {len(remaining)} types")
            try:
                eeg_all = load_cml_eeg_epoched(
                    self.subject, self.experiment, self.session,
                    evs_combined, self.tmin, self.tmax,
                    localization=self.localization, montage=self.montage,
                    scheme=scheme,
                )
                self._vprint(f"  CML bulk EEG shape: {eeg_all.shape}")
                return eeg_all, evs_combined, failed
            except Exception as e:
                self._vprint(f"  Bulk CML load failed: {e}")
                self._vprint(f"  Probing individual types to find the culprit...")

                bad_type = None
                for t in remaining:
                    try:
                        load_cml_eeg_epoched(
                            self.subject, self.experiment, self.session,
                            cml_events_by_type[t], self.tmin, self.tmax,
                            localization=self.localization, montage=self.montage,
                            scheme=scheme,
                        )
                    except Exception:
                        bad_type = t
                        break

                if bad_type is None:
                    self._vprint(f"  Cannot identify bad type; falling back (no bulk)")
                    return None, pd.DataFrame(), failed

                self._vprint(f"  Removing failed type '{bad_type}' and retrying bulk")
                failed.add(bad_type)
                remaining.remove(bad_type)
                status.append((acq, bad_type, "fail", "cml_epoch_out_of_range"))

        return None, pd.DataFrame(), failed

    def _bulk_load_bids(self, bids_events_by_type, types_to_run, status):
        """Attempt bulk BIDS epoch load; on failure, find and remove bad types."""
        acq = self.acq_label
        bids_acq = self.bids_acquisition
        remaining = [t for t in types_to_run if t in bids_events_by_type]
        failed = set()

        while remaining:
            evs_combined = pd.concat(
                [bids_events_by_type[t] for t in remaining],
                ignore_index=True,
            ).drop_duplicates(subset=["sample"], keep="first") \
             .sort_values("sample").reset_index(drop=True)

            self._vprint(f"  Bulk BIDS load: {len(evs_combined)} events, {len(remaining)} types")
            try:
                epochs_all = self.reader.load_epochs(
                    tmin=self.tmin / 1000.0,
                    tmax=self.tmax / 1000.0,
                    events=evs_combined,
                    acquisition=bids_acq,
                    baseline=None,
                    preload=True,
                )

                if self.is_intracranial:
                    picks = mne.pick_types(
                        epochs_all.info, seeg=True, ecog=True,
                        eeg=False, eog=False, misc=False,
                    )
                else:
                    picks = mne.pick_types(
                        epochs_all.info, eeg=True, eog=False, misc=False,
                    )
                if len(picks) == 0:
                    picks = np.arange(len(epochs_all.ch_names))
                epochs_all = epochs_all.pick(picks)

                eeg_all = epochs_to_ptsa(epochs_all, evs_combined)
                eeg_all = convert_unit(eeg_all, "uV", copy=False)
                eeg_all = eeg_all.assign_coords(time=eeg_all["time"] * 1000.0)
                eeg_all["time"].attrs["units"] = "ms"
                self._vprint(f"  BIDS bulk EEG shape: {eeg_all.shape}")
                del epochs_all
                return eeg_all, evs_combined, failed
            except Exception as e:
                self._vprint(f"  Bulk BIDS load failed: {e}")
                self._vprint(f"  Probing individual types to find the culprit...")

                bad_type = None
                for t in remaining:
                    try:
                        self.reader.load_epochs(
                            tmin=self.tmin / 1000.0,
                            tmax=self.tmax / 1000.0,
                            events=bids_events_by_type[t],
                            acquisition=bids_acq,
                            baseline=None,
                            preload=True,
                        )
                    except Exception:
                        bad_type = t
                        break

                if bad_type is None:
                    self._vprint(f"  Cannot identify bad type; falling back (no bulk)")
                    return None, pd.DataFrame(), failed

                self._vprint(f"  Removing failed type '{bad_type}' and retrying bulk")
                failed.add(bad_type)
                remaining.remove(bad_type)
                status.append((acq, bad_type, "fail", "bids_epoch_out_of_range"))

        return None, pd.DataFrame(), failed

    # ==================================================================
    # Main pipeline
    # ==================================================================

    def _run(self) -> Dict[str, Any]:
        acq = self.acq_label
        self._vprint(f"  Acquisition: {acq}")

        self._vprint(f"  Loading CML events...")
        evs_cml = load_cml_events(
            self.subject, self.experiment, self.session,
            localization=self.localization, montage=self.montage,
        )
        self._vprint(f"  CML events loaded: {len(evs_cml)} rows")

        self._vprint(f"  Loading BIDS events (event_type={self.reader.eeg_type})...")
        evs_bids = load_bids_events(self.reader, event_type=self.reader.eeg_type)
        self._vprint(f"  BIDS events loaded: {len(evs_bids)} rows")

        types_to_run = (
            sorted(set(self.evs_types))
            if self.evs_types
            else sorted(evs_cml["type"].dropna().unique())
        )
        self._vprint(f"  Event types to process: {types_to_run}")
        self._vprint(f"  Epoch window: tmin={self.tmin} ms, tmax={self.tmax} ms")

        # CML scheme for iEEG
        scheme = None
        if self.is_intracranial:
            self._vprint(f"  Loading CML contacts and pairs...")
            contacts, pairs = load_cml_contacts_and_pairs(
                self.subject, self.experiment, self.session,
                self.localization, self.montage,
            )
            scheme = contacts if self.acquisition == "contacts" else pairs
            self._vprint(f"  Using scheme: {self.acquisition} ({len(scheme)} channels)")

        comparator = SignalComparator()

        # ----------------------------------------------------------
        # Prepare per-type events (dedupe within each type)
        # ----------------------------------------------------------
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
            f"  Types with CML data: {len(cml_events_by_type)}, "
            f"with BIDS data: {len(bids_events_by_type)}"
        )

        # ----------------------------------------------------------
        # Build sample -> type lookup (for slicing after bulk load)
        # ----------------------------------------------------------
        cml_offsets_by_type = {
            etype: set(evs["eegoffset"].values)
            for etype, evs in cml_events_by_type.items()
        }
        bids_samples_by_type = {
            etype: set(evs["sample"].values)
            for etype, evs in bids_events_by_type.items()
        }

        # ----------------------------------------------------------
        # Bulk load with retry
        # ----------------------------------------------------------
        all_raw, all_summary, all_time, status = [], [], [], []

        eeg_cml_all, evs_cml_combined, failed_cml = self._bulk_load_cml(
            cml_events_by_type, types_to_run, scheme, status,
        )

        eeg_bids_all, evs_bids_combined, failed_bids = self._bulk_load_bids(
            bids_events_by_type, types_to_run, status,
        )

        # ----------------------------------------------------------
        # Per event type: slice from bulk arrays and compare
        # ----------------------------------------------------------
        for etype in types_to_run:
            try:
                self._vprint(f"    Processing event type: {etype}")

                if etype in failed_cml:
                    self._vprint(f"      Already failed (CML bulk load)")
                    continue
                if etype in failed_bids:
                    self._vprint(f"      Already failed (BIDS bulk load)")
                    continue

                if etype not in cml_events_by_type:
                    self._vprint(f"      Skipped: no CML events")
                    status.append((acq, etype, "skip", "no_cml_events"))
                    continue
                if etype not in bids_events_by_type:
                    self._vprint(f"      Skipped: no BIDS events")
                    status.append((acq, etype, "skip", "no_bids_events"))
                    continue

                if eeg_cml_all is None:
                    self._vprint(f"      Skipped: CML bulk load entirely failed")
                    status.append((acq, etype, "fail", "cml_bulk_load_unavailable"))
                    continue
                if eeg_bids_all is None:
                    self._vprint(f"      Skipped: BIDS bulk load entirely failed")
                    status.append((acq, etype, "fail", "bids_bulk_load_unavailable"))
                    continue

                # CML slice
                cml_mask = evs_cml_combined["eegoffset"].isin(cml_offsets_by_type[etype])
                eeg_cml_t = eeg_cml_all.isel(event=np.where(cml_mask.values)[0])
                self._vprint(f"      CML slice: {eeg_cml_t.shape}")

                # BIDS slice
                bids_mask = evs_bids_combined["sample"].isin(bids_samples_by_type[etype])
                eeg_bids_t = eeg_bids_all.isel(event=np.where(bids_mask.values)[0])
                self._vprint(f"      BIDS slice: {eeg_bids_t.shape}")

                # Compare
                self._vprint(f"      Comparing...")
                result = comparator.compare(
                    eeg_bids_t, eeg_cml_t,
                    label_a="BIDS", label_b="CMLReader",
                    subject=self.subject, experiment=self.experiment,
                    session=self.session,
                )
                self._vprint(f"      Done (ok={result.ok})")

                for key, container in [
                    ("df_raw", all_raw),
                    ("df_raw_summary", all_summary),
                    ("df_time", all_time),
                ]:
                    df = result.extras.get(key)
                    if df is not None and not df.empty:
                        df = df.copy()
                        df["event_type"] = etype
                        df["acquisition"] = acq
                        container.append(df)

                status.append((acq, etype, "ok", ""))

            except Exception as e:
                self._vprint(f"      FAILED: {repr(e)}")
                status.append((acq, etype, "fail", repr(e)))

        # ----------------------------------------------------------
        # Cleanup and save
        # ----------------------------------------------------------
        del eeg_cml_all, eeg_bids_all
        gc.collect()

        self._vprint(f"\n  Saving results for {acq}...")
        tag = self.session_tag

        df_raw = pd.concat(all_raw, ignore_index=True) if all_raw else pd.DataFrame()
        df_summary = pd.concat(all_summary, ignore_index=True) if all_summary else pd.DataFrame()
        df_time = pd.concat(all_time, ignore_index=True) if all_time else pd.DataFrame()
        df_status = pd.DataFrame(status, columns=["acquisition", "event_type", "status", "detail"])

        self._save_df(df_raw, f"df_epoch_{tag}_{acq}.csv")
        self._save_df(df_summary, f"df_epoch_summary_{tag}_{acq}.csv")
        self._save_df(df_time, f"df_epoch_time_{tag}_{acq}.csv")
        self._save_df(df_status, f"df_epoch_status_{tag}_{acq}.csv")

        return {
            "df_raw": df_raw,
            "df_raw_summary": df_summary,
            "df_time": df_time,
            "status": df_status,
        }
