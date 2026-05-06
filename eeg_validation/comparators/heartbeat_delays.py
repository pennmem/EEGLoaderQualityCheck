"""HeartbeatDelaysComparator — collect post-correction one-way HEARTBEAT delays.

The histogram produced by ``check_all_heartbeats.ipynb`` plots, for every
HEARTBEAT exchange across many sessions, the residual one-way network delay
that survives the linear task->host clock alignment. This comparator returns
exactly those per-heartbeat residuals as long-form rows so they can be pooled
across sessions and plotted as a histogram.

HEARTBEATs only exist in the CML world (logged on the task laptop's
``session.jsonl`` and the host PC's ``elemem/event.log``). BIDS does not
store them, so this comparator is single-source by construction — there
is no CML-vs-BIDS distinction at the heartbeat-delay level.

For one session it:
  1. Fits ``time_host = slope * time_task + offset`` on clean low-latency
     heartbeats (delegated to ``onsets_heartbeats._fit_clock_alignment``).
  2. Applies the correction to task-side heartbeat times.
  3. Emits one ``df_detail`` row per (count, kind) with ``delay_ms``.
"""

from __future__ import annotations

from typing import List, Optional, Union

import numpy as np
import pandas as pd

from .base import Comparator, ComparisonResult
from .onsets_heartbeats import _fit_clock_alignment
from .utils import one_unique


class HeartbeatDelaysComparator(Comparator):
    """Single-session post-correction HEARTBEAT delay collector.

    Parameters
    ----------
    max_task_latency_ms, max_host_latency_ms : float
        Roundtrip-latency thresholds for picking heartbeats clean enough to
        fit the linear correction on. Defaults match the existing
        ``OnsetsHeartbeatsComparator``.
    min_heartbeats, max_include_heartbeats : int
        Bounds on how many merged heartbeats are used for the linear fit.
    long_delay_ms : float
        Threshold for the "fraction of long delays" tally reported in
        ``df_summary``. Notebook uses 50 ms.
    """

    def __init__(
        self,
        *,
        max_task_latency_ms: float = 2.0,
        max_host_latency_ms: float = 1.0,
        min_heartbeats: int = 180,
        max_include_heartbeats: int = 2000,
        long_delay_ms: float = 50.0,
    ):
        self.max_task_latency_ms = max_task_latency_ms
        self.max_host_latency_ms = max_host_latency_ms
        self.min_heartbeats = min_heartbeats
        self.max_include_heartbeats = max_include_heartbeats
        self.long_delay_ms = long_delay_ms

    def compare(
        self,
        heartbeats: pd.DataFrame,
        b: Optional[pd.DataFrame] = None,
        *,
        subject: Optional[str] = None,
        experiment: Optional[str] = None,
        session: Optional[Union[str, int]] = None,
    ) -> ComparisonResult:
        for col in ("hardware_system", "count", "time"):
            if col not in heartbeats.columns:
                raise ValueError(f"heartbeats missing required column '{col}'")

        subject = subject or one_unique(heartbeats, "subject")
        experiment = experiment or one_unique(heartbeats, "experiment")
        session = session or one_unique(heartbeats, "session")

        fit_error: Optional[str] = None
        try:
            fit = _fit_clock_alignment(
                heartbeats,
                max_task_latency_ms=self.max_task_latency_ms,
                max_host_latency_ms=self.max_host_latency_ms,
                min_heartbeats=self.min_heartbeats,
                max_include_heartbeats=self.max_include_heartbeats,
            )
        except Exception as exc:
            fit_error = f"{type(exc).__name__}: {exc}"
            fit = None

        rows: List[dict] = []
        n_hb = n_hb_ok = 0
        frac_long_hb = frac_long_hb_ok = np.nan

        if fit is not None:
            slope = fit["slope"]
            offset = fit["offset"]

            task = heartbeats[heartbeats["hardware_system"] == "task_laptop"].copy()
            host = heartbeats[heartbeats["hardware_system"] == "host_pc"].copy()

            task["time_corr"] = pd.to_numeric(task["time"], errors="coerce") * slope + offset
            if "time_HEARTBEAT_OK" in task.columns:
                task["time_HEARTBEAT_OK_corr"] = (
                    pd.to_numeric(task["time_HEARTBEAT_OK"], errors="coerce") * slope + offset
                )

            t = task.set_index("count")
            h = host.set_index("count")
            common = t.index.intersection(h.index)
            t = t.loc[common]
            h = h.loc[common]

            hb_delay = (
                pd.to_numeric(h["time"], errors="coerce")
                - pd.to_numeric(t["time_corr"], errors="coerce")
            )
            for cnt, d in hb_delay.dropna().items():
                rows.append({
                    "subject": subject,
                    "experiment": experiment,
                    "session": session,
                    "kind": "HEARTBEAT",
                    "count": int(cnt),
                    "delay_ms": float(d),
                })

            if "time_HEARTBEAT_OK" in t.columns and "time_HEARTBEAT_OK" in h.columns:
                # Notebook plots ``-HEARTBEAT_OK_residual`` so the host->task
                # echo direction shows positive delays. Mirror that here so
                # downstream histograms match the reference plot.
                hb_ok_delay = -(
                    pd.to_numeric(h["time_HEARTBEAT_OK"], errors="coerce")
                    - pd.to_numeric(t["time_HEARTBEAT_OK_corr"], errors="coerce")
                )
                for cnt, d in hb_ok_delay.dropna().items():
                    rows.append({
                        "subject": subject,
                        "experiment": experiment,
                        "session": session,
                        "kind": "HEARTBEAT_OK",
                        "count": int(cnt),
                        "delay_ms": float(d),
                    })

            df_detail = pd.DataFrame(rows)
            if not df_detail.empty:
                hb_vals = df_detail.loc[df_detail["kind"] == "HEARTBEAT", "delay_ms"].to_numpy()
                hb_ok_vals = df_detail.loc[df_detail["kind"] == "HEARTBEAT_OK", "delay_ms"].to_numpy()
                n_hb = int(hb_vals.size)
                n_hb_ok = int(hb_ok_vals.size)
                if n_hb:
                    frac_long_hb = float((np.abs(hb_vals) > self.long_delay_ms).mean())
                if n_hb_ok:
                    frac_long_hb_ok = float((np.abs(hb_ok_vals) > self.long_delay_ms).mean())
        else:
            df_detail = pd.DataFrame(
                columns=["subject", "experiment", "session",
                         "kind", "count", "delay_ms"]
            )

        df_summary = pd.DataFrame([{
            "subject": subject,
            "experiment": experiment,
            "session": session,
            "n_HEARTBEAT": n_hb,
            "n_HEARTBEAT_OK": n_hb_ok,
            "frac_HEARTBEAT_above_long_delay": frac_long_hb,
            "frac_HEARTBEAT_OK_above_long_delay": frac_long_hb_ok,
            "long_delay_threshold_ms": self.long_delay_ms,
            "slope": fit["slope"] if fit is not None else np.nan,
            "offset_ms": fit["offset"] if fit is not None else np.nan,
            "rms_residual_ms": fit["rms_residual_ms"] if fit is not None else np.nan,
            "r2": fit["r2"] if fit is not None else np.nan,
            "average_latency_ms": fit["average_latency"] if fit is not None else np.nan,
            "n_heartbeats_used_for_fit": fit["n_used_for_fit"] if fit is not None else 0,
            "prop_task_lagging_host": fit["prop_task_lagging_host"] if fit is not None else np.nan,
            "fit_error": fit_error,
        }])

        return ComparisonResult(
            ok=fit_error is None,
            df_summary=df_summary,
            df_detail=df_detail,
            df_mismatches=pd.DataFrame(),
            subject=subject,
            experiment=experiment,
            session=session,
            extras={"fit_error": fit_error},
        )
