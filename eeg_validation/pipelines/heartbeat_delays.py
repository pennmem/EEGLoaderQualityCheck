"""HEARTBEAT delays pipeline — System-4 only.

Per-session pipeline that mirrors the EventsPipeline / MontagePipeline
shape: load the source data, run a comparator, save the per-session CSVs
into ``out_path`` so the standard 3e aggregation cell picks them up.

Heartbeats themselves are not stored in either CMLReader output or BIDS;
both pipelines source them from
``/data10/RAM/subjects/<alias>/behavioral/<exp>/session_<sess>/``.
This pipeline therefore loads from ``/data10`` once and emits one
summary row per "source" (cml / bids) so downstream plots can split
sessions by which compare-cml-bids universe they belong to.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import cmlreaders as cml
import numpy as np
import pandas as pd

from .base import BasePipeline
from ..comparators.heartbeat_delays import HeartbeatDelaysComparator
from ..loaders.bids import load_bids_events
from ..loaders.cml import load_cml_events
from ..loaders.heartbeats import load_heartbeats_for_session
from ..preparers.events import prep_events


class HeartbeatDelaysPipeline(BasePipeline):
    """Collect post-correction HEARTBEAT / HEARTBEAT_OK delays for one session.

    Parameters
    ----------
    subject_alias : str, optional
        On-disk subject directory under ``/data10/RAM/subjects/`` (e.g.
        ``R1204T_1`` while ``subject == 'R1204T'``). Defaults to ``subject``.
    original_session : int, optional
        Directory-name session number under
        ``/data10/RAM/subjects/<alias>/behavioral/<exp>/session_<sess>/``.
        Defaults to ``session``.
    long_delay_ms : float
        Threshold for the ``frac_*_above_long_delay`` summary column.
    """

    def __init__(
        self,
        *args,
        subject_alias: Optional[str] = None,
        original_session: Optional[int] = None,
        long_delay_ms: float = 50.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.subject_alias = subject_alias
        self.original_session = original_session
        self.long_delay_ms = long_delay_ms

    def _output_paths(self) -> List[str]:
        return [
            self._make_path("df_heartbeat_summary"),
            self._make_path("df_heartbeat_delays"),
            self._make_path("df_heartbeat_event_deltas"),
        ]

    _EVENT_DELTA_COLUMNS = [
        "subject", "experiment", "session", "trial_type",
        "sample_cml_raw", "sample_bids",
        "mstime_cml_raw", "onset_bids",
        "eegoffset_predicted", "mstime_predicted_ms",
        "delta_sample", "delta_onset_ms",
    ]

    def _compute_event_deltas(
        self, slope: float, offset_ms: float,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Match CML and BIDS events and compute per-event delta columns.

        Returns ``(df_event_deltas, stats)``. ``stats`` includes
        ``n_events_matched``, ``max_abs_delta_sample``,
        ``p99_abs_delta_sample``, ``max_abs_delta_onset_ms``,
        ``p99_abs_delta_onset_ms``, and ``event_delta_error``. On any
        failure mode (missing columns, row-count mismatch, load
        exception) ``df_event_deltas`` is empty and ``event_delta_error``
        is set; this never raises so a single bad session won't abort a
        batch.
        """
        empty = pd.DataFrame(columns=self._EVENT_DELTA_COLUMNS)
        stats: Dict[str, Any] = {
            "n_events_matched": 0,
            "max_abs_delta_sample": np.nan,
            "p99_abs_delta_sample": np.nan,
            "max_abs_delta_onset_ms": np.nan,
            "p99_abs_delta_onset_ms": np.nan,
            "event_delta_error": None,
        }

        try:
            reader_cml = cml.CMLReader(
                subject=self.subject,
                experiment=self.experiment,
                session=self.session,
                localization=self.localization,
                montage=self.montage,
            )
            evs_cml = load_cml_events(
                self.subject, self.experiment, self.session,
                localization=self.localization, montage=self.montage,
            )
            evs_cml = cml.correct_retrieval_offsets(evs_cml, reader_cml)
            evs_cml = cml.correct_countdown_lists(evs_cml, reader_cml)
        except Exception as e:
            stats["event_delta_error"] = f"cml_events_load_failed: {e}"
            self._vprint(f"  events: CML load failed ({e})")
            return empty, stats

        try:
            evs_bids = load_bids_events(
                reader=self.reader, event_type=self.reader.device,
            )
        except Exception as e:
            stats["event_delta_error"] = f"bids_events_load_failed: {e}"
            self._vprint(f"  events: BIDS load failed ({e})")
            return empty, stats

        try:
            prep = prep_events(
                evs_cml, evs_bids,
                onset_as_diff=False,
                subject=self.subject,
                experiment=self.experiment,
                session=self.session,
            )
        except Exception as e:
            stats["event_delta_error"] = f"prep_events_failed: {e}"
            self._vprint(f"  events: prep_events failed ({e})")
            return empty, stats

        cml_p = prep["evs_cml_prepped"]
        bids_p = prep["evs_bids_prepped"]
        if len(cml_p) != len(bids_p):
            stats["event_delta_error"] = (
                f"row_count_mismatch: cml={len(cml_p)} bids={len(bids_p)}"
            )
            self._vprint(
                f"  events: row count mismatch (cml={len(cml_p)}, "
                f"bids={len(bids_p)})"
            )
            return empty, stats
        if len(cml_p) == 0:
            stats["event_delta_error"] = "no_events_after_prep"
            return empty, stats

        # ``prep_events`` renames CML eegoffset->sample and overwrites
        # mstime->onset (in seconds, anchored). For the delta math we need
        # the original CML eegoffset / mstime aligned to the prepped row
        # order; reattach them by index since prep_events filters but
        # preserves row order via the index.
        eegoffset_raw = (
            pd.to_numeric(evs_cml.loc[cml_p.index, "eegoffset"], errors="coerce")
            .to_numpy(dtype=float)
        )
        mstime_raw = (
            pd.to_numeric(evs_cml.loc[cml_p.index, "mstime"], errors="coerce")
            .to_numpy(dtype=float)
        )
        sample_bids = (
            pd.to_numeric(evs_bids.loc[bids_p.index, "sample"], errors="coerce")
            .to_numpy(dtype=float)
        )
        onset_bids = (
            pd.to_numeric(evs_bids.loc[bids_p.index, "onset"], errors="coerce")
            .to_numpy(dtype=float)
        )

        # `mstime_raw`/`eegoffset_raw` are already on the host/EEG clock
        # (eegoffset is samples since EEGSTART). The heartbeat fit is
        # `time_host = slope*time_task + offset_ms`, so its constant `offset_ms`
        # (the ~9 h task<->host skew) must be SUBTRACTED via the inverse fit, never
        # added — adding it pushes mstime ~9 h onto the wrong clock and the derived
        # eegoffset tens of millions of samples negative.
        #
        # eegoffset (samples) is slope-fitted; mstime is the continuous inverse fit
        # onto the task clock, task = (host - offset_ms)/slope (no quantization,
        # inter-event spacing scaled by 1/slope per the fit).
        eegoffset_pred = np.round(slope * eegoffset_raw)
        mstime_pred = (mstime_raw - offset_ms) / slope

        # Anchor both sides at the first matched event so the unknown
        # recording-start offset between absolute mstime and
        # BIDS-relative onset cancels out.
        valid = (
            np.isfinite(eegoffset_pred)
            & np.isfinite(mstime_pred)
            & np.isfinite(sample_bids)
            & np.isfinite(onset_bids)
        )
        if not valid.any():
            stats["event_delta_error"] = "no_finite_rows"
            return empty, stats
        first = int(np.argmax(valid))
        delta_sample = (sample_bids - sample_bids[first]) - (
            eegoffset_pred - eegoffset_pred[first]
        )
        delta_onset_ms = (onset_bids * 1000.0 - onset_bids[first] * 1000.0) - (
            mstime_pred - mstime_pred[first]
        )

        df = pd.DataFrame({
            "subject": self.subject,
            "experiment": self.experiment,
            "session": self.session,
            "trial_type": cml_p["trial_type"].to_numpy(),
            "sample_cml_raw": eegoffset_raw,
            "sample_bids": sample_bids,
            "mstime_cml_raw": mstime_raw,
            "onset_bids": onset_bids,
            "eegoffset_predicted": eegoffset_pred,
            "mstime_predicted_ms": mstime_pred,
            "delta_sample": delta_sample,
            "delta_onset_ms": delta_onset_ms,
        }, columns=self._EVENT_DELTA_COLUMNS)

        finite_ds = df["delta_sample"].dropna().to_numpy()
        finite_dt = df["delta_onset_ms"].dropna().to_numpy()
        stats.update({
            "n_events_matched": int(valid.sum()),
            "max_abs_delta_sample": float(np.max(np.abs(finite_ds))) if finite_ds.size else np.nan,
            "p99_abs_delta_sample": float(np.quantile(np.abs(finite_ds), 0.99)) if finite_ds.size else np.nan,
            "max_abs_delta_onset_ms": float(np.max(np.abs(finite_dt))) if finite_dt.size else np.nan,
            "p99_abs_delta_onset_ms": float(np.quantile(np.abs(finite_dt), 0.99)) if finite_dt.size else np.nan,
        })
        return df, stats

    def _bids_events_present(self) -> bool:
        device = self.reader.device or "ieeg"
        suffix = "ieeg" if device == "ieeg" else "eeg"
        path = os.path.join(
            self.bids_root,
            f"sub-{self.subject}",
            f"ses-{self.session}",
            suffix,
            f"sub-{self.subject}_ses-{self.session}_task-{self.experiment}_events.tsv",
        )
        return os.path.exists(path)

    def _run(self) -> Dict[str, Any]:
        self._vprint("  Loading heartbeats from /data10...")
        try:
            hb = load_heartbeats_for_session(
                self.subject, self.experiment, int(self.session),
                subject_alias=self.subject_alias,
                original_session=self.original_session,
            )
        except Exception as e:
            self._vprint(f"  Skipped: heartbeats not available ({e})")
            return {"skipped": True, "reason": "heartbeats_not_found", "error": str(e)}

        n_task = int((hb["hardware_system"] == "task_laptop").sum())
        n_host = int((hb["hardware_system"] == "host_pc").sum())
        self._vprint(f"  Heartbeats loaded: task={n_task}, host={n_host}")

        bids_ok = self._bids_events_present()
        self._vprint(f"  BIDS events present: {bids_ok}")

        # ---- Heartbeat fit + per-heartbeat delays (CML-only by nature) ----
        cmp = HeartbeatDelaysComparator(long_delay_ms=self.long_delay_ms)
        res = cmp.compare(
            hb,
            subject=self.subject,
            experiment=self.experiment,
            session=self.session,
        )
        self._vprint(
            f"  heartbeat delays: ok={res.ok} "
            f"n_HB={int(res.df_summary.iloc[0]['n_HEARTBEAT'])}"
        )
        df_sum = res.df_summary.copy()
        df_del = (
            res.df_detail.copy() if not res.df_detail.empty
            else pd.DataFrame(columns=[
                "subject", "experiment", "session",
                "kind", "count", "delay_ms",
            ])
        )
        fit_error = df_sum.iloc[0]["fit_error"]

        # ---- Per-event CML→BIDS deltas anchored on the heartbeat fit ----
        df_event_deltas = pd.DataFrame(columns=self._EVENT_DELTA_COLUMNS)
        delta_stats: Dict[str, Any] = {
            "n_events_matched": 0,
            "max_abs_delta_sample": np.nan,
            "p99_abs_delta_sample": np.nan,
            "max_abs_delta_onset_ms": np.nan,
            "p99_abs_delta_onset_ms": np.nan,
            "event_delta_error": None,
        }
        if not bids_ok:
            delta_stats["event_delta_error"] = "bids_events_not_found"
        elif fit_error:
            delta_stats["event_delta_error"] = f"clock_fit_failed: {fit_error}"
        else:
            slope = float(df_sum.iloc[0]["slope"])
            offset_ms = float(df_sum.iloc[0]["offset_ms"])
            df_event_deltas, delta_stats = self._compute_event_deltas(
                slope=slope, offset_ms=offset_ms,
            )
            self._vprint(
                f"  event deltas: n={delta_stats['n_events_matched']} "
                f"|Δsample|max={delta_stats['max_abs_delta_sample']} "
                f"|Δonset_ms|p99={delta_stats['p99_abs_delta_onset_ms']} "
                f"err={delta_stats['event_delta_error']}"
            )

        # Attach event-delta stats to the (single) summary row.
        for col, val in delta_stats.items():
            df_sum[col] = val
        df_sum["bids_present"] = bool(bids_ok)

        self._save_df(df_sum, f"df_heartbeat_summary_{self.session_tag}.csv")
        self._save_df(df_del, f"df_heartbeat_delays_{self.session_tag}.csv")
        self._save_df(
            df_event_deltas,
            f"df_heartbeat_event_deltas_{self.session_tag}.csv",
        )

        return {
            "result": res,
            "paths": self._output_paths(),
            "bids_present": bids_ok,
            "n_HEARTBEAT": int(df_sum.iloc[0]["n_HEARTBEAT"]),
            "fit_error": fit_error,
            "event_deltas": delta_stats,
        }
