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

from bidsreader import CMLBIDSReader


class EpochedPipeline(BasePipeline):
    """Compare CML vs BIDS epoched EEG for one session, per event type.

    Loads ALL epochs once per acquisition stream, then slices per event
    type for comparison.  If the bulk load fails, it identifies which
    event types caused the failure and retries without them.

    For each event type:
      1. CML: filter events -> dedupe by eegoffset -> slice from bulk array
      2. BIDS: filter events -> dedupe by sample  -> slice from bulk array
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
        tag = self.session_tag
        for acq in acqs:
            for prefix in ("df_epoch_summary", "df_epoch_status"):
                paths.append(os.path.join(self.out_path, f"{prefix}_{tag}_{acq}.csv"))
        return paths

    # ==================================================================
    # Bulk load with retry helpers
    # ==================================================================

    def _bulk_load_cml(self, cml_events_by_type, types_to_run, scheme, status, acq_tag):
        """Attempt bulk CML epoch load; on failure, find and remove bad types.

        Returns
        -------
        eeg_all : xarray or None
            Bulk-loaded EEG (None if everything failed).
        evs_combined : pd.DataFrame
            Combined events used for the successful load.
        failed : set
            Event types that were removed due to load failures.
        """
        remaining = [t for t in types_to_run if t in cml_events_by_type]
        failed = set()

        while remaining:
            evs_combined = pd.concat(
                [cml_events_by_type[t] for t in remaining],
                ignore_index=True,
            ).drop_duplicates(subset=["eegoffset"], keep="first") \
             .sort_values("eegoffset").reset_index(drop=True)

            self._vprint(f"  Bulk CML load: {len(evs_combined)} events, types={remaining}")
            try:
                eeg_all = load_cml_eeg_epoched(
                    self.subject, self.experiment, self.session,
                    evs_combined, self.tmin, self.tmax,
                    localization=self.localization, montage=self.montage,
                    scheme=scheme,
                )
                eeg_all = self.cml_to_volts(eeg_all)
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
                status.append((acq_tag, bad_type, "fail", "cml_epoch_out_of_range"))

        return None, pd.DataFrame(), failed

    def _bulk_load_bids(self, bids_events_by_type, types_to_run, bids_acq, status, acq_tag):
        """Attempt bulk BIDS epoch load; on failure, find and remove bad types.

        Returns
        -------
        eeg_all : xarray or None
            Bulk-loaded EEG (None if everything failed).
        evs_combined : pd.DataFrame
            Combined events used for the successful load.
        failed : set
            Event types that were removed due to load failures.
        """
        remaining = [t for t in types_to_run if t in bids_events_by_type]
        failed = set()

        while remaining:
            evs_combined = pd.concat(
                [bids_events_by_type[t] for t in remaining],
                ignore_index=True,
            ).drop_duplicates(subset=["sample"], keep="first") \
             .sort_values("sample").reset_index(drop=True)

            self._vprint(f"  Bulk BIDS load: {len(evs_combined)} events, types={remaining}")
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
                # epochs_all = convert_unit(epochs_all, "uV", copy=False)
                # epochs_all._data *= 1e6
                # epochs_all._data = np.round(epochs_all._data)
                self._vprint(epochs_all._data)

                # MNE silently drops epochs that extend past the recording.
                # Filter evs_combined to only the events MNE kept so that
                # the event dimension matches epochs.get_data().shape[0].
                kept = epochs_all.selection
                if len(kept) < len(evs_combined):
                    n_dropped = len(evs_combined) - len(kept)
                    self._vprint(f"  MNE dropped {n_dropped} epoch(s) outside recording bounds")
                    evs_combined = evs_combined.iloc[kept].reset_index(drop=True)

                eeg_all = epochs_to_ptsa(epochs_all, evs_combined)

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
                status.append((acq_tag, bad_type, "fail", "bids_epoch_out_of_range"))

        return None, pd.DataFrame(), failed

    # ==================================================================
    # Main pipeline
    # ==================================================================

    def _run(self) -> Dict[str, Any]:
        self._vprint(f"  Loading CML events...")
        evs_cml = load_cml_events(
            self.subject, self.experiment, self.session,
            localization=self.localization, montage=self.montage,
        )
        self._vprint(f"  CML events loaded: {len(evs_cml)} rows")

        self._vprint(f"  Loading BIDS events (event_type={self.reader.device})...")
        evs_bids = load_bids_events(self.reader, event_type=self.reader.device)
        self._vprint(f"  BIDS events loaded: {len(evs_bids)} rows")

        types_to_run = (
            sorted(set(self.evs_types))
            if self.evs_types
            else sorted(evs_cml["type"].dropna().unique())
        )
        self._vprint(f"  Event types to process: {types_to_run}")
        self._vprint(f"  Epoch window: tmin={self.tmin} ms, tmax={self.tmax} ms")

        cml_schemes = {}
        if self.is_intracranial:
            self._vprint(f"  Loading CML contacts and pairs...")
            contacts, pairs = load_cml_contacts_and_pairs(
                self.subject, self.experiment, self.session,
                self.localization, self.montage,
            )
            cml_schemes = {"monopolar": contacts, "bipolar": pairs}
            self._vprint(f"  Contacts: {len(contacts)}, Pairs: {len(pairs)}")

        acq_tags = ["monopolar", "bipolar"] if self.is_intracranial else ["eeg"]
        self._vprint(f"  Acquisition streams: {acq_tags}")

        comparator = SignalComparator()
        results_all = {}

        for acq_tag in acq_tags:
            self._vprint(f"\n  === Acquisition: {acq_tag} ===")
            bids_acq = acq_tag if self.is_intracranial else None
            scheme = cml_schemes.get(acq_tag)

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
                f"  Types with CML data: {sorted(cml_events_by_type.keys())}\n"
                f"  Types with BIDS data: {sorted(bids_events_by_type.keys())}"
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
            all_summary, status = [], []

            eeg_cml_all, evs_cml_combined, failed_cml = self._bulk_load_cml(
                cml_events_by_type, types_to_run, scheme, status, acq_tag,
            )

            eeg_bids_all, evs_bids_combined, failed_bids = self._bulk_load_bids(
                bids_events_by_type, types_to_run, bids_acq, status, acq_tag,
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
                        status.append((acq_tag, etype, "skip", "no_cml_events"))
                        continue
                    if etype not in bids_events_by_type:
                        self._vprint(f"      Skipped: no BIDS events")
                        status.append((acq_tag, etype, "skip", "no_bids_events"))
                        continue

                    if eeg_cml_all is None:
                        self._vprint(f"      Skipped: CML bulk load entirely failed")
                        status.append((acq_tag, etype, "fail", "cml_bulk_load_unavailable"))
                        continue
                    if eeg_bids_all is None:
                        self._vprint(f"      Skipped: BIDS bulk load entirely failed")
                        status.append((acq_tag, etype, "fail", "bids_bulk_load_unavailable"))
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
                        acquisition=acq_tag,
                    )
                    self._vprint(f"      Done (ok={result.ok})")

                    df_sum = result.extras.get("df_raw_summary")
                    if df_sum is not None and not df_sum.empty:
                        df_sum = df_sum.copy()
                        df_sum["event_type"] = etype
                        df_sum["acquisition"] = acq_tag
                        all_summary.append(df_sum)

                    status.append((acq_tag, etype, "ok", ""))

                except Exception as e:
                    self._vprint(f"      FAILED: {repr(e)}")
                    status.append((acq_tag, etype, "fail", repr(e)))

            # ----------------------------------------------------------
            # All-events-combined comparison
            # ----------------------------------------------------------
            if eeg_cml_all is not None and eeg_bids_all is not None:
                try:
                    self._vprint(f"    Processing ALL event types combined")
                    result = comparator.compare(
                        eeg_bids_all, eeg_cml_all,
                        label_a="BIDS", label_b="CMLReader",
                        subject=self.subject, experiment=self.experiment,
                        session=self.session,
                        acquisition=acq_tag,
                    )
                    df = result.extras.get("df_raw_summary")
                    if df is not None and not df.empty:
                        df = df.copy()
                        df["event_type"] = "ALL"
                        all_summary.append(df)
                    status.append((acq_tag, "ALL", "ok", ""))
                except Exception as e:
                    self._vprint(f"      ALL combined FAILED: {repr(e)}")
                    status.append((acq_tag, "ALL", "fail", repr(e)))

            # ----------------------------------------------------------
            # Cleanup and save
            # ----------------------------------------------------------
            del eeg_cml_all, eeg_bids_all
            gc.collect()

            self._vprint(f"\n  Saving results for {acq_tag}...")
            tag = self.session_tag

            df_summary = pd.concat(all_summary, ignore_index=True) if all_summary else pd.DataFrame()
            df_status = pd.DataFrame(status, columns=["acquisition", "event_type", "status", "detail"])

            self._save_df(df_summary, f"df_epoch_summary_{tag}_{acq_tag}.csv")
            self._save_df(df_status, f"df_epoch_status_{tag}_{acq_tag}.csv")

            results_all[acq_tag] = {
                "df_epoch_summary": df_summary,
                "df_epoch_status": df_status,
            }

        return results_all