"""
OnsetsHeartbeatsComparator — validate event onsets against the clock alignment
implied by task-laptop / host-PC HEARTBEAT exchanges (System 4).

Background
----------
On every System-4 session the task laptop sends a HEARTBEAT to Elemem on the
host PC every ~1 s; the host echoes back a HEARTBEAT_OK. Both sides log the
event with a monotonic counter and a local timestamp, giving two independent
clocks observing the same tick stream.

This comparator wraps the analysis from ``check_all_heartbeats.ipynb`` into a
single ``ComparisonResult``. It quantifies, for one session:

- Heartbeat health: roundtrip latency distribution, percentage of pathological
  roundtrips, regularity of the 1 Hz cadence.
- Clock alignment: the linear ``time_host = slope * time_task + offset`` fit
  used to map task-side event timestamps onto the host-side EEG sample axis,
  plus its R^2 and RMS residual.
- Post-correction residuals: the per-heartbeat one-way delays that survive
  the linear correction (the practical jitter floor on any event timestamp).
- Per-event impact: applying the fitted correction to the supplied event
  onsets, the maximum eegoffset shift and the slope-induced mstime drift
  across the session.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .base import Comparator, ComparisonResult
from .utils import one_unique


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _summarize_latencies(latencies: pd.Series, *, high_latency_ms: float) -> Dict[str, float]:
    """Per-session latency stats, mirroring the verbose block in get_heart()."""
    lat = pd.to_numeric(latencies, errors="coerce").dropna()
    if len(lat) == 0:
        return {
            "n": 0,
            "min": np.nan, "max": np.nan,
            "p10": np.nan, "p50": np.nan, "p90": np.nan,
            "mean": np.nan,
            "frac_high_latency": np.nan,
        }
    return {
        "n": int(len(lat)),
        "min": float(lat.min()),
        "max": float(lat.max()),
        "p10": float(lat.quantile(0.10)),
        "p50": float(lat.quantile(0.50)),
        "p90": float(lat.quantile(0.90)),
        "mean": float(lat.mean()),
        "frac_high_latency": float((lat > high_latency_ms).mean()),
    }


def _heartbeat_spacing_stats(
    times_ms: pd.Series,
    *,
    min_spacing_ms: float,
    max_spacing_ms: float,
) -> Dict[str, float]:
    """Quantify how regular the 1 Hz HEARTBEAT cadence was on the task side."""
    diffs = pd.to_numeric(times_ms, errors="coerce").diff().dropna()
    if len(diffs) == 0:
        return {
            "n_intervals": 0,
            "frac_out_of_range": np.nan,
            "min_spacing_ms": np.nan,
            "max_spacing_ms": np.nan,
            "mean_spacing_ms": np.nan,
        }
    out_of_range = (diffs < min_spacing_ms) | (diffs > max_spacing_ms)
    return {
        "n_intervals": int(len(diffs)),
        "frac_out_of_range": float(out_of_range.mean()),
        "min_spacing_ms": float(diffs.min()),
        "max_spacing_ms": float(diffs.max()),
        "mean_spacing_ms": float(diffs.mean()),
    }


def _fit_clock_alignment(
    heartbeats: pd.DataFrame,
    *,
    max_task_latency_ms: float,
    max_host_latency_ms: float,
    min_heartbeats: int,
    max_include_heartbeats: int,
) -> Dict[str, Any]:
    """Fit ``time_host = slope * time_task + offset`` from clean heartbeats.

    Mirrors ``get_heartbeat_correction`` in the notebook: filters to low-latency
    pairs, merges on ``count``, fits a linear regression, and adjusts the
    intercept by half the average roundtrip.
    """
    task = heartbeats[
        (heartbeats["hardware_system"] == "task_laptop")
        & (pd.to_numeric(heartbeats["latency"], errors="coerce") < max_task_latency_ms)
    ]
    host = heartbeats[
        (heartbeats["hardware_system"] == "host_pc")
        & (pd.to_numeric(heartbeats["latency"], errors="coerce") < max_host_latency_ms)
    ]
    merged = pd.merge(task, host, on="count", suffixes=("_task", "_host"))
    merged = merged.dropna(subset=["time_task", "time_host", "latency_task", "latency_host"])

    n_used_total = len(merged)
    if n_used_total < min_heartbeats:
        raise ValueError(
            f"Available low-latency HEARTBEATs ({n_used_total}) < min_heartbeats ({min_heartbeats})"
        )

    if n_used_total > max_include_heartbeats:
        half = max_include_heartbeats // 2
        merged = pd.concat([merged.iloc[:half], merged.iloc[-half:]])

    t_task = pd.to_numeric(merged["time_task"], errors="coerce").to_numpy(dtype=float)
    t_host = pd.to_numeric(merged["time_host"], errors="coerce").to_numpy(dtype=float)

    # Closed-form least squares: slope, intercept, R^2.
    t_task_mean = t_task.mean()
    t_host_mean = t_host.mean()
    cov = np.mean((t_task - t_task_mean) * (t_host - t_host_mean))
    var = np.mean((t_task - t_task_mean) ** 2)
    if var == 0:
        raise ValueError("Degenerate clock fit: task-side timestamps have zero variance")
    slope = float(cov / var)
    offset = float(t_host_mean - slope * t_task_mean)

    predicted = slope * t_task + offset
    residuals = t_host - predicted
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((t_host - t_host_mean) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    rms_residual = float(np.sqrt(ss_res / len(t_host)))

    avg_latency = float(pd.to_numeric(merged["latency_task"], errors="coerce").mean())
    latency_correction = avg_latency / 2.0
    adjusted_offset = offset - latency_correction

    # Notebook's sanity check: after subtracting the half-roundtrip, residuals
    # implying the host received a tick before the task sent it are physically
    # impossible.
    prop_task_lagging_host = float(((-residuals - latency_correction) > 0).mean())

    return {
        "slope": slope,
        "uncorrected_offset": offset,
        "offset": adjusted_offset,
        "average_latency": avg_latency,
        "r2": r2,
        "rms_residual_ms": rms_residual,
        "prop_task_lagging_host": prop_task_lagging_host,
        "n_used_for_fit": int(len(merged)),
        "n_eligible": int(n_used_total),
        "merged_for_fit": merged,
        "residuals_ms": residuals,
    }


def _apply_correction(
    events: pd.DataFrame,
    *,
    slope: float,
    offset: float,
    time_col: str,
    sample_col: Optional[str],
) -> pd.DataFrame:
    """Apply ``t -> t * slope + offset`` to ``time_col`` and ``s -> s * slope`` to ``sample_col``.

    Mirrors the notebook's ``correct_event_times`` plus the per-event eegoffset
    rounding step from the example demo cell.
    """
    out = events.copy()
    if time_col in out.columns:
        out[time_col] = pd.to_numeric(out[time_col], errors="coerce") * slope + offset
    if sample_col is not None and sample_col in out.columns:
        original_dtype = events[sample_col].dtype
        scaled = pd.to_numeric(out[sample_col], errors="coerce") * slope
        if pd.api.types.is_integer_dtype(original_dtype):
            scaled = scaled.round().astype(original_dtype, errors="ignore")
        out[sample_col] = scaled
    return out


# ----------------------------------------------------------------------
# Comparator
# ----------------------------------------------------------------------

class OnsetsHeartbeatsComparator(Comparator):
    """Validate event onsets against heartbeat-derived clock alignment.

    Inputs
    ------
    a : events DataFrame
        Must contain a millisecond timestamp column (``mstime`` by default)
        and optionally an integer sample column (``eegoffset`` by default).
    b : heartbeats DataFrame
        Long-form, as produced by ``get_heart`` in
        ``check_all_heartbeats.ipynb``: one row per HEARTBEAT, with rows for
        both ``hardware_system in {'task_laptop', 'host_pc'}`` for each
        ``count``. Required columns: ``hardware_system``, ``count``, ``time``,
        ``latency``.

    Tolerances
    ----------
    max_rms_residual_ms : float
        Largest acceptable RMS of the host-vs-fitted-task residual (notebook
        default 2 ms). Above this the verdict is ``ok=False``.
    max_slope_deviation : float
        Largest acceptable ``|slope - 1|``. Notebook uses ``1e-5``.
    max_p90_latency_ms : float
        Largest acceptable 90th-percentile roundtrip latency (default 5 ms).
    max_frac_high_latency : float
        Largest acceptable fraction of roundtrips above ``high_latency_ms``.
    max_frac_spacing_out_of_range : float
        Largest acceptable fraction of HEARTBEAT intervals outside
        ``[min_spacing_ms, max_spacing_ms]``.

    Filtering / fitting
    -------------------
    max_task_latency_ms, max_host_latency_ms : float
        Roundtrip thresholds for picking heartbeats clean enough to fit on.
    min_heartbeats, max_include_heartbeats : int
        Bounds on how many merged heartbeats are used for the linear fit.
    min_spacing_ms, max_spacing_ms : float
        Acceptable HEARTBEAT-to-HEARTBEAT cadence on the task side.
    high_latency_ms : float
        Threshold for the "% high latency" tally (notebook uses 100 ms).
    long_delay_ms : float
        Threshold for the "% long one-way delay" tally on residuals
        (notebook uses 50 ms).

    Per-event reporting
    -------------------
    max_event_eegoffset_shift_samples : float
        Per-event sample-shift above which an event is flagged into
        ``df_mismatches``. Defaults to 2 samples.
    """

    DEFAULT_TIME_COL = "mstime"
    DEFAULT_SAMPLE_COL = "eegoffset"

    def __init__(
        self,
        *,
        # Verdict thresholds.
        max_rms_residual_ms: float = 2.0,
        max_slope_deviation: float = 1e-5,
        max_p90_latency_ms: float = 5.0,
        max_frac_high_latency: float = 0.01,
        max_frac_spacing_out_of_range: float = 0.005,
        # Heartbeat filtering for fit.
        max_task_latency_ms: float = 2.0,
        max_host_latency_ms: float = 1.0,
        min_heartbeats: int = 180,
        max_include_heartbeats: int = 2000,
        # Reporting thresholds.
        min_spacing_ms: float = 990.0,
        max_spacing_ms: float = 1010.0,
        high_latency_ms: float = 100.0,
        long_delay_ms: float = 50.0,
        # Per-event flagging.
        max_event_eegoffset_shift_samples: float = 2.0,
        max_event_mstime_shift_ms: float = 5.0,
        max_mismatches: int = 50,
        # Drift visualization reference.
        sfreq_ref_hz: float = 1000.0,
        session_duration_ref_s: float = 3600.0,
    ):
        self.max_rms_residual_ms = max_rms_residual_ms
        self.max_slope_deviation = max_slope_deviation
        self.max_p90_latency_ms = max_p90_latency_ms
        self.max_frac_high_latency = max_frac_high_latency
        self.max_frac_spacing_out_of_range = max_frac_spacing_out_of_range
        self.max_task_latency_ms = max_task_latency_ms
        self.max_host_latency_ms = max_host_latency_ms
        self.min_heartbeats = min_heartbeats
        self.max_include_heartbeats = max_include_heartbeats
        self.min_spacing_ms = min_spacing_ms
        self.max_spacing_ms = max_spacing_ms
        self.high_latency_ms = high_latency_ms
        self.long_delay_ms = long_delay_ms
        self.max_event_eegoffset_shift_samples = max_event_eegoffset_shift_samples
        self.max_event_mstime_shift_ms = max_event_mstime_shift_ms
        self.max_mismatches = max_mismatches
        self.sfreq_ref_hz = sfreq_ref_hz
        self.session_duration_ref_s = session_duration_ref_s

    # ------------------------------------------------------------------
    def compare(
        self,
        events: pd.DataFrame,
        heartbeats: pd.DataFrame,
        *,
        label_a: str = "events",
        label_b: str = "heartbeats",
        time_col: Optional[str] = None,
        sample_col: Optional[str] = None,
        subject: Optional[str] = None,
        experiment: Optional[str] = None,
        session: Optional[Union[str, int]] = None,
        return_aligned: bool = False,
    ) -> ComparisonResult:
        time_col = time_col or self.DEFAULT_TIME_COL
        sample_col = sample_col or self.DEFAULT_SAMPLE_COL

        for col in ("hardware_system", "count", "time", "latency"):
            if col not in heartbeats.columns:
                raise ValueError(f"heartbeats missing required column '{col}'")
        if time_col not in events.columns:
            raise ValueError(f"events missing required time column '{time_col}'")

        subject = subject or one_unique(events, "subject") or one_unique(heartbeats, "subject")
        experiment = experiment or one_unique(events, "experiment") or one_unique(heartbeats, "experiment")
        session = session or one_unique(events, "session") or one_unique(heartbeats, "session")

        task_hb = heartbeats[heartbeats["hardware_system"] == "task_laptop"]
        host_hb = heartbeats[heartbeats["hardware_system"] == "host_pc"]

        # ---- Heartbeat health ---------------------------------------
        latency_stats = _summarize_latencies(
            task_hb["latency"], high_latency_ms=self.high_latency_ms
        )
        spacing_stats = _heartbeat_spacing_stats(
            task_hb.sort_values("count")["time"],
            min_spacing_ms=self.min_spacing_ms,
            max_spacing_ms=self.max_spacing_ms,
        )

        # ---- Clock alignment ----------------------------------------
        fit_failed: Optional[str] = None
        try:
            fit = _fit_clock_alignment(
                heartbeats,
                max_task_latency_ms=self.max_task_latency_ms,
                max_host_latency_ms=self.max_host_latency_ms,
                min_heartbeats=self.min_heartbeats,
                max_include_heartbeats=self.max_include_heartbeats,
            )
        except Exception as exc:
            fit_failed = f"{type(exc).__name__}: {exc}"
            fit = {
                "slope": np.nan,
                "uncorrected_offset": np.nan,
                "offset": np.nan,
                "average_latency": np.nan,
                "r2": np.nan,
                "rms_residual_ms": np.nan,
                "prop_task_lagging_host": np.nan,
                "n_used_for_fit": 0,
                "n_eligible": 0,
                "merged_for_fit": pd.DataFrame(),
                "residuals_ms": np.array([]),
            }

        # ---- Post-correction one-way residuals ----------------------
        residuals_HEARTBEAT_ms = np.array([])
        residuals_HEARTBEAT_OK_ms = np.array([])
        frac_long_delay_HEARTBEAT = np.nan
        frac_long_delay_HEARTBEAT_OK = np.nan

        if fit_failed is None:
            slope = fit["slope"]
            offset = fit["offset"]

            task_full = task_hb.copy()
            task_full["time_corrected"] = (
                pd.to_numeric(task_full["time"], errors="coerce") * slope + offset
            )
            if "time_HEARTBEAT_OK" in task_full.columns:
                task_full["time_HEARTBEAT_OK_corrected"] = (
                    pd.to_numeric(task_full["time_HEARTBEAT_OK"], errors="coerce") * slope + offset
                )

            t_task = task_full.set_index("count")
            t_host = host_hb.set_index("count")
            common_count = t_task.index.intersection(t_host.index)
            t_task_c = t_task.loc[common_count]
            t_host_c = t_host.loc[common_count]

            residuals_HEARTBEAT_ms = (
                pd.to_numeric(t_host_c["time"], errors="coerce")
                - pd.to_numeric(t_task_c["time_corrected"], errors="coerce")
            ).dropna().to_numpy(dtype=float)

            if "time_HEARTBEAT_OK" in t_task_c.columns and "time_HEARTBEAT_OK" in t_host_c.columns:
                residuals_HEARTBEAT_OK_ms = (
                    pd.to_numeric(t_host_c["time_HEARTBEAT_OK"], errors="coerce")
                    - pd.to_numeric(t_task_c["time_HEARTBEAT_OK_corrected"], errors="coerce")
                ).dropna().to_numpy(dtype=float)

            if residuals_HEARTBEAT_ms.size:
                frac_long_delay_HEARTBEAT = float(
                    (np.abs(residuals_HEARTBEAT_ms) > self.long_delay_ms).mean()
                )
            if residuals_HEARTBEAT_OK_ms.size:
                frac_long_delay_HEARTBEAT_OK = float(
                    (np.abs(residuals_HEARTBEAT_OK_ms) > self.long_delay_ms).mean()
                )

        # ---- Per-event impact of the correction ---------------------
        events_corrected: Optional[pd.DataFrame] = None
        delta_time = np.array([])
        delta_sample = np.array([])
        max_delta_sample = np.nan
        max_drift_ms = np.nan
        constant_delta_ms = np.nan
        mismatch_rows: List[Dict[str, Any]] = []

        if fit_failed is None and len(events) > 0:
            events_corrected = _apply_correction(
                events,
                slope=fit["slope"],
                offset=fit["offset"],
                time_col=time_col,
                sample_col=sample_col,
            )
            t_orig = pd.to_numeric(events[time_col], errors="coerce").to_numpy(dtype=float)
            t_corr = pd.to_numeric(events_corrected[time_col], errors="coerce").to_numpy(dtype=float)
            delta_time = t_corr - t_orig

            if delta_time.size:
                constant_delta_ms = float(np.nanmin(np.abs(delta_time)))
                # The constant component of the shift is the offset; the
                # interesting part is the slope-induced drift across the
                # session, which is delta_time minus its first finite value.
                first_valid = next((v for v in delta_time if np.isfinite(v)), np.nan)
                if np.isfinite(first_valid):
                    drift_ms = delta_time - first_valid
                    max_drift_ms = float(np.nanmax(np.abs(drift_ms)))

            if sample_col in events.columns:
                s_orig = pd.to_numeric(events[sample_col], errors="coerce").to_numpy(dtype=float)
                s_corr = pd.to_numeric(events_corrected[sample_col], errors="coerce").to_numpy(dtype=float)
                delta_sample = s_corr - s_orig
                if delta_sample.size:
                    max_delta_sample = float(np.nanmax(np.abs(delta_sample)))

            # Flag events whose shift exceeds the per-event tolerances.
            for i, (dt_ms, ds_samples) in enumerate(
                zip(
                    delta_time,
                    delta_sample if delta_sample.size else np.full_like(delta_time, np.nan),
                )
            ):
                # Skip the constant offset component on the time-shift check —
                # we care about session-relative drift, not the offset itself.
                drift_ms_event = (
                    dt_ms - (np.nanmin(np.abs(delta_time)) if delta_time.size else 0.0)
                )
                ds_bad = (
                    np.isfinite(ds_samples)
                    and abs(ds_samples) > self.max_event_eegoffset_shift_samples
                )
                dt_bad = (
                    np.isfinite(drift_ms_event)
                    and abs(drift_ms_event) > self.max_event_mstime_shift_ms
                )
                if ds_bad or dt_bad:
                    mismatch_rows.append({
                        "subject": subject,
                        "experiment": experiment,
                        "session": session,
                        "i": int(i),
                        f"{label_a}_{time_col}_orig": float(t_orig[i]) if delta_time.size else np.nan,
                        f"{label_a}_{time_col}_corrected": float(t_corr[i]) if delta_time.size else np.nan,
                        "delta_time_ms": float(dt_ms) if np.isfinite(dt_ms) else np.nan,
                        "drift_ms_relative_to_session_start": float(drift_ms_event)
                        if np.isfinite(drift_ms_event) else np.nan,
                        "delta_sample": float(ds_samples) if np.isfinite(ds_samples) else np.nan,
                    })
                    if len(mismatch_rows) >= self.max_mismatches:
                        break

        # ---- Implied drift over a reference 1-hour session ----------
        if np.isfinite(fit["slope"]):
            n_samples_ref = self.session_duration_ref_s * self.sfreq_ref_hz
            drift_samples_1hr = (fit["slope"] - 1.0) * n_samples_ref
            drift_ms_1hr = (fit["slope"] - 1.0) * self.session_duration_ref_s * 1000.0
        else:
            drift_samples_1hr = np.nan
            drift_ms_1hr = np.nan

        # ---- Build verdict ------------------------------------------
        verdict_failures: List[str] = []
        if fit_failed is not None:
            verdict_failures.append(f"clock fit failed ({fit_failed})")
        else:
            if fit["rms_residual_ms"] > self.max_rms_residual_ms:
                verdict_failures.append(
                    f"rms_residual {fit['rms_residual_ms']:.3f} ms > {self.max_rms_residual_ms} ms"
                )
            if abs(fit["slope"] - 1.0) > self.max_slope_deviation:
                verdict_failures.append(
                    f"|slope - 1| {abs(fit['slope'] - 1.0):.2e} > {self.max_slope_deviation}"
                )
        if (
            np.isfinite(latency_stats["p90"])
            and latency_stats["p90"] > self.max_p90_latency_ms
        ):
            verdict_failures.append(
                f"p90 latency {latency_stats['p90']:.2f} ms > {self.max_p90_latency_ms} ms"
            )
        if (
            np.isfinite(latency_stats["frac_high_latency"])
            and latency_stats["frac_high_latency"] > self.max_frac_high_latency
        ):
            verdict_failures.append(
                f"fraction of latencies > {self.high_latency_ms} ms = "
                f"{latency_stats['frac_high_latency']:.4f} > {self.max_frac_high_latency}"
            )
        if (
            np.isfinite(spacing_stats["frac_out_of_range"])
            and spacing_stats["frac_out_of_range"] > self.max_frac_spacing_out_of_range
        ):
            verdict_failures.append(
                f"fraction of out-of-range HEARTBEAT spacings = "
                f"{spacing_stats['frac_out_of_range']:.4f} > {self.max_frac_spacing_out_of_range}"
            )

        ok = len(verdict_failures) == 0

        # ---- Build summary frame ------------------------------------
        summary_row = {
            "subject": subject,
            "experiment": experiment,
            "session": session,
            "comparison": f"{label_a} vs {label_b}",
            "ok": bool(ok),
            "verdict_failures": verdict_failures,
            # Heartbeat-side health
            "n_heartbeats_task": int(len(task_hb)),
            "n_heartbeats_host": int(len(host_hb)),
            "n_heartbeats_used_for_fit": int(fit["n_used_for_fit"]),
            "latency_min_ms": latency_stats["min"],
            "latency_p10_ms": latency_stats["p10"],
            "latency_p50_ms": latency_stats["p50"],
            "latency_p90_ms": latency_stats["p90"],
            "latency_max_ms": latency_stats["max"],
            "latency_mean_ms": latency_stats["mean"],
            "frac_latency_above_threshold": latency_stats["frac_high_latency"],
            "high_latency_threshold_ms": self.high_latency_ms,
            "spacing_min_ms": spacing_stats["min_spacing_ms"],
            "spacing_max_ms": spacing_stats["max_spacing_ms"],
            "spacing_mean_ms": spacing_stats["mean_spacing_ms"],
            "frac_spacing_out_of_range": spacing_stats["frac_out_of_range"],
            # Clock alignment fit
            "slope": fit["slope"],
            "offset_ms": fit["offset"],
            "uncorrected_offset_ms": fit["uncorrected_offset"],
            "average_roundtrip_latency_ms": fit["average_latency"],
            "r2": fit["r2"],
            "rms_residual_ms": fit["rms_residual_ms"],
            "prop_task_lagging_host_after_correction": fit["prop_task_lagging_host"],
            # Post-correction one-way residual histogram summaries
            "n_residuals_HEARTBEAT": int(residuals_HEARTBEAT_ms.size),
            "n_residuals_HEARTBEAT_OK": int(residuals_HEARTBEAT_OK_ms.size),
            "frac_HEARTBEAT_residual_above_long_delay": frac_long_delay_HEARTBEAT,
            "frac_HEARTBEAT_OK_residual_above_long_delay": frac_long_delay_HEARTBEAT_OK,
            "long_delay_threshold_ms": self.long_delay_ms,
            # Drift implication
            "implied_drift_samples_per_ref_hour": drift_samples_1hr,
            "implied_drift_ms_per_ref_hour": drift_ms_1hr,
            "drift_ref_sfreq_hz": self.sfreq_ref_hz,
            "drift_ref_session_seconds": self.session_duration_ref_s,
            # Per-event impact
            "n_events": int(len(events)),
            "constant_event_delta_ms": constant_delta_ms,
            "max_event_drift_ms": max_drift_ms,
            "max_event_eegoffset_shift_samples": max_delta_sample,
            "n_events_flagged": int(len(mismatch_rows)),
            "fit_error": fit_failed,
        }
        df_summary = pd.DataFrame([summary_row])

        # ---- Build per-event detail frame ---------------------------
        if events_corrected is not None and delta_time.size:
            df_detail = pd.DataFrame({
                "subject": subject,
                "experiment": experiment,
                "session": session,
                f"{time_col}_orig": pd.to_numeric(events[time_col], errors="coerce").to_numpy(),
                f"{time_col}_corrected": pd.to_numeric(events_corrected[time_col], errors="coerce").to_numpy(),
                "delta_time_ms": delta_time,
            })
            if delta_sample.size:
                df_detail[f"{sample_col}_orig"] = pd.to_numeric(
                    events[sample_col], errors="coerce"
                ).to_numpy()
                df_detail[f"{sample_col}_corrected"] = pd.to_numeric(
                    events_corrected[sample_col], errors="coerce"
                ).to_numpy()
                df_detail["delta_sample"] = delta_sample
        else:
            df_detail = pd.DataFrame()

        df_mismatches = pd.DataFrame(mismatch_rows)

        result = ComparisonResult(
            ok=ok,
            df_summary=df_summary,
            df_detail=df_detail,
            df_mismatches=df_mismatches,
            subject=subject,
            experiment=experiment,
            session=session,
            extras={
                "fit": {k: v for k, v in fit.items() if k not in ("merged_for_fit", "residuals_ms")},
                "fit_residuals_ms": fit["residuals_ms"],
                "merged_heartbeats_for_fit": fit["merged_for_fit"],
                "HEARTBEAT_residuals_ms": residuals_HEARTBEAT_ms,
                "HEARTBEAT_OK_residuals_ms": residuals_HEARTBEAT_OK_ms,
                "verdict_failures": verdict_failures,
            },
        )

        if return_aligned and events_corrected is not None:
            result.aligned_a = events.reset_index(drop=True)
            result.aligned_b = events_corrected.reset_index(drop=True)

        return result
