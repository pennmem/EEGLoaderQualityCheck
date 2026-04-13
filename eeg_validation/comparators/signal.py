"""
SignalComparator — pairwise comparison of EEG xarray DataArrays.

Combines the old ``compare_raw_signal_pairs``, ``compare_time_coord_pairs``,
and ``compare_eeg_sources`` into one class.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import xarray as xr

from .base import Comparator, ComparisonResult
from .utils import crop_to_min_length


# ======================================================================
# Helpers (previously module-level functions)
# ======================================================================

def strip_event_metadata(da: xr.DataArray) -> xr.DataArray:
    """Drop event-dim coordinates except 'event' itself."""
    if "event" not in da.dims:
        return da
    drop = [c for c in da.coords if "event" in da.coords[c].dims and c != "event"]
    return da.drop_vars(drop) if drop else da


def ensure_dims(
    da: xr.DataArray,
    event_dim: str = "event",
    channel_dim: str = "channel",
    time_dim: str = "time",
) -> xr.DataArray:
    """Guarantee (event, channel, time) ordering, adding a singleton event dim if needed."""
    if event_dim not in da.dims:
        da = da.expand_dims({event_dim: [0]})
    for d in (event_dim, channel_dim, time_dim):
        if d not in da.dims:
            raise ValueError(f"Expected dim '{d}' not found. Have dims={da.dims}")
    return da.transpose(event_dim, channel_dim, time_dim)


def _affine_residual_mse(a: np.ndarray, b: np.ndarray, fallback_mse: float) -> float:
    """Per-channel MSE after fitting ``a ≈ m*b + c`` by least squares.

    The structural BDF / EDF gain+offset floor between two readers of the
    same recording is *exactly* an affine transform of the underlying
    samples. Subtracting the best-fit affine reconstruction isolates the
    residual disagreement that no calibration choice can explain.

    Returns ``fallback_mse`` if the fit cannot be performed (too few finite
    samples, or B is constant).
    """
    a_flat = np.asarray(a, dtype=float).ravel()
    b_flat = np.asarray(b, dtype=float).ravel()
    mask = np.isfinite(a_flat) & np.isfinite(b_flat)
    if mask.sum() < 2:
        return fallback_mse
    am = a_flat[mask]
    bm = b_flat[mask]
    if not np.any(bm != bm[0]):  # b is constant -> fit is degenerate
        return fallback_mse
    M = np.vstack([bm, np.ones_like(bm)]).T
    try:
        (slope, intercept), *_ = np.linalg.lstsq(M, am, rcond=None)
    except np.linalg.LinAlgError:
        return fallback_mse
    residual = am - (slope * bm + intercept)
    return float(np.mean(residual ** 2))


def channel_overlap_summary(
    a: xr.DataArray, b: xr.DataArray, channel_dim: str = "channel"
) -> Dict[str, Any]:
    """Return overlap stats between two DataArrays on the channel dim."""
    a_ch = list(a[channel_dim].values) if channel_dim in a.dims else []
    b_ch = list(b[channel_dim].values) if channel_dim in b.dims else []
    set_a, set_b = set(map(str, a_ch)), set(map(str, b_ch))
    common = [ch for ch in a_ch if str(ch) in set_b]
    return {
        "n_a": len(a_ch),
        "n_b": len(b_ch),
        "n_common": len(common),
        "only_a": sorted(set_a - set_b),
        "only_b": sorted(set_b - set_a),
        "common": common,
        "order_matches": (
            [ch for ch in a_ch if str(ch) in set_b]
            == [ch for ch in b_ch if str(ch) in set_a]
        ),
    }


# ======================================================================
# SignalComparator
# ======================================================================

class SignalComparator(Comparator):
    """Compare two EEG DataArrays (raw or epoched) channel-by-channel.

    Handles:
    - Raw signals  (channel × time per event)
    - Time coordinate alignment
    - Pairwise channel statistics

    Notes on the BDF / EDF microvolt floor
    --------------------------------------
    EDF/BDF stores per-channel ``physical_min`` and ``physical_max`` as
    8-character ASCII fields, so any reader of a BDF-backed dataset
    reconstructs samples through a gain that is rounded relative to the
    "true" calibration. When two pipelines load the same recording through
    two independently-derived gains (e.g. CMLReader applies a sidecar
    calibration to int16 split files while a BIDS BDF was re-encoded with
    its own header), the per-sample disagreement is dominated by an
    *affine* term ``a ≈ m * b + c``. To distinguish this structural floor
    from real disagreement we per-channel fit a least-squares line
    ``a = m*b + c`` and report the residual energy as ``mse_excess``.
    ``mse_excess ≈ 0`` means the two readers agree up to a linear gain
    + offset transformation; ``mse_excess > 0`` indicates real
    sample-by-sample disagreement that no calibration choice can explain.

    Parameters
    ----------
    rtol, atol : float
        Tolerance for ``np.isclose``.
    max_mismatches_per_channel : int
        Max mismatch indices to report per channel.
    strip_metadata : bool
        Drop event-dim metadata coordinates before comparing.
    """

    def __init__(
        self,
        *,
        rtol: float = 1e-6,
        atol: float = 1e-9,
        max_mismatches_per_channel: int = 10,
        strip_metadata: bool = True,
    ):
        self.rtol = rtol
        self.atol = atol
        self.max_mismatches_per_channel = max_mismatches_per_channel
        self.strip_metadata = strip_metadata

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def compare(
        self,
        da_a: xr.DataArray,
        da_b: xr.DataArray,
        *,
        label_a: str = "A",
        label_b: str = "B",
        subject: Optional[str] = None,
        experiment: Optional[str] = None,
        session: Optional[Union[str, int]] = None,
        acquisition: Optional[str] = None,
        compare_signals: bool = True,
        compare_time: bool = True,
    ) -> ComparisonResult:
        """Run signal and/or time-coordinate comparison."""
        # Standardize
        a = self._prepare(da_a)
        b = self._prepare(da_b)

        # Determine acquisition tag: "scalp" for scalp EEG, otherwise
        # pass through the value (e.g. "monopolar", "bipolar").
        if acquisition is None or acquisition.lower() in ("eeg", "scalp"):
            acq_tag = "scalp"
        else:
            acq_tag = acquisition  # "monopolar" or "bipolar"

        extras: Dict[str, Any] = {}
        extras["channel_overlap"] = channel_overlap_summary(a, b)

        frames_signal = []
        frames_signal_summary = []
        frames_time = []

        if compare_signals:
            df_sig, df_sig_summary = self._compare_signals(
                a, b, label_a, label_b, subject, experiment, session, acq_tag
            )
            frames_signal.append(df_sig)
            frames_signal_summary.append(df_sig_summary)

        if compare_time:
            df_time = self._compare_time(
                a, b, label_a, label_b, subject, experiment, session, acq_tag
            )
            frames_time.append(df_time)

        df_raw = pd.concat(frames_signal, ignore_index=True) if frames_signal else pd.DataFrame()
        df_raw_summary = pd.concat(frames_signal_summary, ignore_index=True) if frames_signal_summary else pd.DataFrame()
        df_time = pd.concat(frames_time, ignore_index=True) if frames_time else pd.DataFrame()

        any_mismatch = False
        if len(df_raw_summary):
            any_mismatch |= bool(df_raw_summary["n_close_diff_channels"].sum() > 0)
        if len(df_time):
            any_mismatch |= bool(df_time["n_close_time_diff"].sum() > 0)

        return ComparisonResult(
            ok=not any_mismatch,
            df_summary=df_raw_summary,
            df_detail=df_raw,
            df_mismatches=df_time,  # time stats as "mismatches" slot
            subject=subject,
            experiment=experiment,
            session=session,
            extras={
                "df_raw": df_raw,
                "df_raw_summary": df_raw_summary,
                "df_time": df_time,
                **extras,
            },
        )

    # ------------------------------------------------------------------
    # Compare raw signals (per channel × event)
    # ------------------------------------------------------------------
    def _compare_signals(
        self, da_a, da_b, label_a, label_b, subject, experiment, session, acq_tag="scalp"
    ):
        common_ch = np.intersect1d(
            da_a["channel"].astype(str).values,
            da_b["channel"].astype(str).values,
        )
        n_events = int(min(da_a.sizes.get("event", 1), da_b.sizes.get("event", 1)))
        any_close_by_event = np.zeros(n_events, dtype=bool)

        rows = []
        exact_fail = []
        close_fail = []

        # Per-channel aggregates aligned with `common_ch`.
        channel_means: List[float] = []
        channel_stds: List[float] = []
        channel_mse: List[float] = []
        channel_mse_excess: List[float] = []

        for ch in common_ch:
            a = np.squeeze(np.asarray(da_a.sel(channel=ch).data))
            b = np.squeeze(np.asarray(da_b.sel(channel=ch).data))

            if a.ndim == 1:
                a = a[None, :]
            if b.ndim == 1:
                b = b[None, :]

            E = min(a.shape[0], b.shape[0], n_events)
            a, b = a[:E], b[:E]
            a2, b2, m = crop_to_min_length(a, b)

            both_nan = np.isnan(a2) & np.isnan(b2)
            exact_bad = ~((a2 == b2) | both_nan)
            close_bad = ~np.isclose(a2, b2, rtol=self.rtol, atol=self.atol, equal_nan=True)

            if np.any(exact_bad):
                exact_fail.append(str(ch))
            if np.any(close_bad):
                close_fail.append(str(ch))

            any_close_by_event[:E] |= np.any(close_bad, axis=1)

            diff = np.where(both_nan | np.isnan(a2) | np.isnan(b2), np.nan, a2 - b2)
            abs_diff = np.abs(diff)

            # Per-channel mean / std of the B (BIDS) signal across all events × time.
            channel_means.append(float(np.nanmean(b2)))
            channel_stds.append(float(np.nanstd(b2)))

            # Per-channel mse and the affine-residual mse_excess.
            ch_mse = float(np.nanmean(diff ** 2))
            channel_mse.append(ch_mse)
            channel_mse_excess.append(_affine_residual_mse(a2, b2, ch_mse))

            for ev_i in range(E):
                rows.append({
                    "subject": subject,
                    "experiment": experiment,
                    "session": session,
                    "acquisition": acq_tag,
                    "comparison": f"{label_a} vs {label_b}",
                    "channel": str(ch),
                    "event": int(ev_i),
                    "n_close_diff": int(close_bad[ev_i].sum()),
                    "mean_abs_diff": float(np.nanmean(abs_diff[ev_i])),
                    "max_abs_diff": float(np.nanmax(abs_diff[ev_i])),
                    "mean_signed_diff": float(np.nanmean(diff[ev_i])),
                    "std_diff": float(np.nanstd(diff[ev_i])),
                    "mse_channel": float(np.nanmean(diff[ev_i] ** 2)),
                    "time_compared_samples": int(m),
                })

        df_detail = pd.DataFrame(rows)

        close_diff_events = np.where(any_close_by_event)[0].tolist()

        def _safe(fn, series):
            vals = pd.to_numeric(series, errors="coerce")
            return float(fn(vals)) if np.isfinite(vals).any() else np.nan

        # Scalar mse_excess: mean across channels of the affine-residual mse.
        if channel_mse_excess:
            mse_excess_scalar = float(np.nanmean(channel_mse_excess))
        else:
            mse_excess_scalar = np.nan

        df_summary = pd.DataFrame([{
            "subject": subject,
            "experiment": experiment,
            "session": session,
            "comparison": f"{label_a} vs {label_b}",
            "source_a": label_a,
            "source_b": label_b,
            "n_common_channels": len(common_ch),
            "n_exact_diff_channels": len(exact_fail),
            "n_close_diff_channels": len(close_fail),
            "mean_abs_diff": _safe(np.nanmean, df_detail["mean_abs_diff"]) if len(df_detail) else np.nan,
            "max_abs_diff": _safe(np.nanmax, df_detail["max_abs_diff"]) if len(df_detail) else np.nan,
            "mean_signed_diff": _safe(np.nanmean, df_detail["mean_signed_diff"]) if len(df_detail) else np.nan,
            "std_diff": _safe(np.nanmean, df_detail["std_diff"]) if len(df_detail) else np.nan,
            "mse": _safe(np.nanmean, df_detail["mse_channel"]) if len(df_detail) else np.nan,
            "mse_excess": mse_excess_scalar,
            "n_events": n_events,
            "n_close_diff_events": len(close_diff_events),
            "close_diff_event_indices": close_diff_events,
            "exact_diff_channels": exact_fail,
            "close_diff_channels": close_fail,
            "common_channels": [str(c) for c in common_ch],
            "channel_means": channel_means,
            "channel_stds": channel_stds,
            "channel_mse": channel_mse,
            "channel_mse_excess": channel_mse_excess,
            "acquisition": acq_tag
        }])

        return df_detail, df_summary

    # ------------------------------------------------------------------
    # Compare time coordinates
    # ------------------------------------------------------------------
    def _compare_time(
        self, da_a, da_b, label_a, label_b, subject, experiment, session, acq_tag="scalp"
    ):
        t_a = np.asarray(da_a["time"].values if "time" in da_a.coords else np.arange(da_a.sizes["time"], dtype=float))
        t_b = np.asarray(da_b["time"].values if "time" in da_b.coords else np.arange(da_b.sizes["time"], dtype=float))

        n_events = int(min(da_a.sizes.get("event", 1), da_b.sizes.get("event", 1)))

        def _to_2d(t):
            t = np.asarray(t)
            if t.ndim == 1:
                return np.tile(t[None, :], (n_events, 1))
            return t[:n_events]

        Ta, Tb = _to_2d(t_a), _to_2d(t_b)
        m = min(Ta.shape[1], Tb.shape[1])
        Ta, Tb = Ta[:, :m], Tb[:, :m]

        both_nan = np.isnan(Ta) & np.isnan(Tb)
        close_bad = ~np.isclose(Ta, Tb, rtol=self.rtol, atol=self.atol, equal_nan=True)
        exact_bad = ~((Ta == Tb) | both_nan)

        diff = np.where(both_nan | np.isnan(Ta) | np.isnan(Tb), np.nan, Ta - Tb)
        abs_diff = np.abs(diff)

        close_by_event = np.any(close_bad, axis=1)

        return pd.DataFrame([{
            "subject": subject,
            "experiment": experiment,
            "session": session,
            "acquisition": acq_tag,
            "comparison": f"{label_a} vs {label_b}",
            "time_len_a": int(da_a.sizes["time"]),
            "time_len_b": int(da_b.sizes["time"]),
            "time_compared_samples": int(m),
            "n_events": n_events,
            "n_close_diff_events": int(close_by_event.sum()),
            "close_diff_event_indices": np.where(close_by_event)[0].tolist(),
            "n_exact_time_diff": int(np.sum(exact_bad)),
            "n_close_time_diff": int(np.sum(close_bad)),
            "mean_abs_time_diff": float(np.nanmean(abs_diff)) if np.isfinite(abs_diff).any() else np.nan,
            "max_abs_time_diff": float(np.nanmax(abs_diff)) if np.isfinite(abs_diff).any() else np.nan,
            "mean_signed_time_diff": float(np.nanmean(diff)) if np.isfinite(diff).any() else np.nan,
            "std_time_diff": float(np.nanstd(diff)) if np.isfinite(diff).any() else np.nan,
            "mse_time": float(np.nanmean(diff ** 2)) if np.isfinite(diff).any() else np.nan,
        }])

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _prepare(self, da: xr.DataArray) -> xr.DataArray:
        if self.strip_metadata:
            da = strip_event_metadata(da)
        return ensure_dims(da)
