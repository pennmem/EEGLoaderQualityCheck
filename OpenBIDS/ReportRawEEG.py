import numpy as np
import xarray as xr
import pandas as pd

# ----------------------------
# USER SETTINGS
# ----------------------------
RTOL = 1e-6
ATOL = 1e-9
MAX_MISMATCHES_PER_CHANNEL = 10

PRINT_FULL_RAW_IF_SMALL = True
MAX_ELEMENTS_FOR_FULL_PRINT = 50_000
PRINT_SLICE = True
SLICE_CFG = {"event": 0, "channel": slice(0, 5), "time": slice(0, 20)}

# ----------------------------
# HELPERS
# ----------------------------
def load_bdf_as_xarray(path: str, *, event_dim_name="event") -> xr.DataArray:
    f = pyedflib.EdfReader(path)
    try:
        n_channels = f.signals_in_file
        ch_names = list(f.getSignalLabels())
        sfreq = float(f.getSampleFrequency(0))
        n_samples = int(f.getNSamples()[0])

        # read + convert each channel explicitly to volts
        data_V = []
        for ch in range(n_channels):
            # print(ch)
            x = f.readSignal(ch)
            unit = f.getPhysicalDimension(ch).lower()

            if unit in ("uv", "µv"):
                x = x * 1e-6          # µV → V
            elif unit == "mv":
                x = x * 1e-3          # mV → V
            elif unit == "v":
                pass                 # already volts
            else:
                print(f"Unknown or invalid physical unit '{unit}' \n" + f"for channel {ch} ({ch_names[ch]})")

            data_V.append(x)

        # shape: (channel, time)
        data = np.vstack(data_V)

        times = np.arange(n_samples) / sfreq

        # add singleton event dimension
        data = data[None, :, :]  # (event=1, channel, time)

        da = xr.DataArray(
            data,
            dims=(event_dim_name, "channel", "time"),
            coords={
                event_dim_name: [0],
                "channel": ch_names,
                "time": times,
                "samplerate": sfreq,
            },
            name="eeg",
            attrs={
                "units": "V",
                "source": "pyedflib (explicitly converted to volts)",
            },
        )

        return da

    finally:
        f.close()
        
    
def strip_event_metadata(da: xr.DataArray) -> xr.DataArray:
    if "event" not in da.dims:
        return da
    drop = [c for c in da.coords if ("event" in da.coords[c].dims and c != "event")]
    return da.drop_vars(drop) if drop else da


def summarize_channel_coords(a: xr.DataArray, b: xr.DataArray, channel_dim: str):
    a_ch = list(a[channel_dim].values) if channel_dim in a.dims else []
    b_ch = list(b[channel_dim].values) if channel_dim in b.dims else []

    set_a, set_b = set(a_ch), set(b_ch)
    only_a = sorted(set_a - set_b)
    only_b = sorted(set_b - set_a)
    common = [ch for ch in a_ch if ch in set_b]  # preserve a's order for common

    print("\n--- Channel coord check ---")
    print("BIDS #channels:", len(a_ch))
    print("CML  #channels:", len(b_ch))
    print("Common channels:", len(common))
    if only_a:
        print("Only in BIDS (first 20):", only_a[:20])
    if only_b:
        print("Only in CML  (first 20):", only_b[:20])

    # order check on common channels
    b_common_in_b_order = [ch for ch in b_ch if ch in set_a]
    if common != b_common_in_b_order:
        print("Channel order differs across objects (on common channels).")
    else:
        print("Channel order matches (on common channels).")

    return a_ch, b_ch, common


def report_mismatched_channels(channel_name, da_a: xr.DataArray, da_b: xr.DataArray, rtol: float, atol: float, max_n: int):
    """
    Prints first max_n mismatch locations for:
      - exact mismatch (NaN-safe)
      - allclose mismatch (rtol/atol, NaN-safe)

    Works for 1D or ND arrays.
    """
    a = np.asarray(da_a.data)
    b = np.asarray(da_b.data)

    # squeeze singleton dims so (1, T) -> (T,)
    a = np.squeeze(a)
    b = np.squeeze(b)

    if a.shape != b.shape:
        print(f"\n[{channel_name}] cannot report mismatches: shape bids {a.shape} vs cml {b.shape}")
        return

    both_nan = np.isnan(a) & np.isnan(b)
    exact_bad = ~((a == b) | both_nan)
    close_bad = ~np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=True)

    def _print_bad(mask, label):
        if not np.any(mask):
            return
        print(f"\n[{channel_name}] {label} mismatches (up to {max_n})")
        inds = np.argwhere(mask)

        for idx_arr in inds[:max_n]:
            idx = tuple(idx_arr.tolist())

            # coords: try to print time (common case)
            coord_info = {}
            if da_a.ndim == 1 and "time" in da_a.dims:
                # idx is (time_index,)
                ti = idx[0]
                try:
                    coord_info["time"] = float(da_a["time"].values[ti])
                except Exception:
                    coord_info["time_index"] = ti
            else:
                # ND case: map each dim
                for d, i in zip(da_a.dims, idx):
                    if d in da_a.coords and da_a.coords[d].ndim == 1:
                        try:
                            coord_info[d] = da_a.coords[d].values[i]
                        except Exception:
                            coord_info[f"{d}_index"] = i
                    else:
                        coord_info[f"{d}_index"] = i

            aval = a[idx]
            bval = b[idx]
            if label == "ALLCLOSE":
                err = np.abs(aval - bval)
                print(" index:", idx, "coords:", coord_info, "bids:", aval, "cml:", bval, "abs_err:", err)
            else:
                print(" index:", idx, "coords:", coord_info, "bids:", aval, "cml:", bval)

    # _print_bad(exact_bad, "EXACT")
    # _print_bad(close_bad, "ALLCLOSE")


def print_raw_data(name: str, da: xr.DataArray):
    data = da.data
    n_elem = data.size

    print(f"\n--- {name} raw data ---")
    print("dims:", da.dims)
    print("shape:", da.shape)

    if PRINT_FULL_RAW_IF_SMALL and n_elem <= MAX_ELEMENTS_FOR_FULL_PRINT:
        print(data)
        return

    if PRINT_SLICE:
        sel = {dim: spec for dim, spec in SLICE_CFG.items() if dim in da.dims}
        try:
            sliced = da.isel(**sel)
            print(f"(showing slice {sel})")
            print(sliced.data)
        except Exception as e:
            print(f"(could not slice with {sel}: {e})")
            flat = data.ravel()
            print("(printing first 200 flattened values)")
            print(flat[: min(200, flat.size)])
    else:
        print("(raw data printing disabled; set PRINT_SLICE or PRINT_FULL_RAW_IF_SMALL)")

def _ensure_dims(da: xr.DataArray, *, event_dim="event", channel_dim="channel", time_dim="time") -> xr.DataArray:
    if event_dim not in da.dims:
        da = da.expand_dims({event_dim: [0]})
    for d in (event_dim, channel_dim, time_dim):
        if d not in da.dims:
            raise ValueError(f"Expected dim '{d}' not found. Have dims={da.dims}")
    return da.transpose(event_dim, channel_dim, time_dim)

def _crop_time_to_min(a: np.ndarray, b: np.ndarray):
    """
    Crop two arrays along the last axis to the min length.
    Returns cropped arrays and min_len.
    """
    la = a.shape[-1]
    lb = b.shape[-1]
    m = min(la, lb)
    return a[..., :m], b[..., :m], m

# def compare_raw_signal_pairs(label_a: str, da_a: xr.DataArray, label_b: str, da_b: xr.DataArray):
#     import pandas as pd
#     rows = []
#     exact_fail = []
#     close_fail = []

#     # Only compare channels that exist in BOTH
#     common_ch = np.intersect1d(
#         da_a["channel"].astype(str).values,
#         da_b["channel"].astype(str).values,
#     )

#     for ch in common_ch:
#         da1 = da_a.sel({"channel": ch})
#         da2 = da_b.sel({"channel": ch})

#         a = np.squeeze(np.asarray(da1.data))  # typically (time,)
#         b = np.squeeze(np.asarray(da2.data))

#         # Handle arbitrary shapes; compare on last axis as time
#         try:
#             a2, b2, m = _crop_time_to_min(a, b)
#         except Exception:
#             exact_fail.append(ch)
#             close_fail.append(ch)
#             rows.append(dict(
#                 comparison=f"{label_a} vs {label_b}",
#                 channel=ch,
#                 n_exact_diff=np.nan,
#                 n_close_diff=np.nan,
#                 mean_abs_diff=np.nan,
#                 max_abs_diff=np.nan,
#                 mean_signed_diff=np.nan,
#                 std_diff=np.nan,
#                 mse_channel=np.nan,
#                 note="shape/crop failure",
#             ))
#             continue

#         both_nan = np.isnan(a2) & np.isnan(b2)
#         exact_bad = ~((a2 == b2) | both_nan)
#         close_bad = ~np.isclose(a2, b2, rtol=RTOL, atol=ATOL, equal_nan=True)

#         diff = a2 - b2
#         invalid = both_nan | np.isnan(a2) | np.isnan(b2)
#         diff = np.where(invalid, np.nan, diff)
#         abs_diff = np.abs(diff)

#         n_exact = int(np.sum(exact_bad))
#         n_close = int(np.sum(close_bad))
#         mean_abs = float(np.nanmean(abs_diff)) if np.isfinite(abs_diff).any() else np.nan
#         max_abs  = float(np.nanmax(abs_diff))  if np.isfinite(abs_diff).any() else np.nan
#         mean_signed = float(np.nanmean(diff)) if np.isfinite(diff).any() else np.nan
#         std_diff = float(np.nanstd(diff)) if np.isfinite(diff).any() else np.nan
#         mse_channel = float(np.nanmean(diff**2)) if np.isfinite(diff).any() else np.nan

#         rows.append(dict(
#             comparison=f"{label_a} vs {label_b}",
#             channel=ch,
#             n_exact_diff=n_exact,
#             n_close_diff=n_close,
#             mean_abs_diff=mean_abs,
#             max_abs_diff=max_abs,
#             mean_signed_diff=mean_signed,
#             std_diff=std_diff,
#             mse_channel=mse_channel,
#             time_compared_samples=int(m),
#         ))

#         if n_exact != 0:
#             exact_fail.append(ch)
#         if n_close != 0:
#             close_fail.append(ch)

#         if (n_exact != 0) or (n_close != 0):
#             # For reporting: make 1D "time" views (event singleton -> isel(0))
#             da1_rep = da1.isel(event=0) if ("event" in da1.dims and da1.sizes.get("event", 0) == 1) else da1
#             da2_rep = da2.isel(event=0) if ("event" in da2.dims and da2.sizes.get("event", 0) == 1) else da2

#             # Crop the DataArrays for mismatch reporting too (by index)
#             # Use isel on time dimension so we don't touch coords.
#             if "time" in da1_rep.dims and "time" in da2_rep.dims:
#                 m2 = min(da1_rep.sizes["time"], da2_rep.sizes["time"])
#                 da1_rep = da1_rep.isel(time=slice(0, m2))
#                 da2_rep = da2_rep.isel(time=slice(0, m2))

#             report_mismatched_channels(
#                 f"{ch} [{label_a} vs {label_b}]",
#                 da1_rep,
#                 da2_rep,
#                 RTOL,
#                 ATOL,
#                 MAX_MISMATCHES_PER_CHANNEL
#             )

#     df = pd.DataFrame(rows)
#     return df, exact_fail, close_fail, common_ch

def compare_raw_signal_pairs(label_a: str, da_a: xr.DataArray, label_b: str, da_b: xr.DataArray):
    """
    Now supports:
      - raw:   (event=1, channel, time) or (channel, time) after _ensure_dims()
      - epoched: (event>1, channel, time)

    Returns:
      df (rows per channel x event),
      exact_fail_channels (kept for backward compat; channel has ANY exact mismatch in ANY event),
      close_fail_channels (channel has ANY close mismatch in ANY event),
      common_ch,
      close_diff_event_indices (events where ANY channel mismatches close),
      n_events
    """
    rows = []
    exact_fail = []
    close_fail = []

    # Only compare channels that exist in BOTH
    common_ch = np.intersect1d(
        da_a["channel"].astype(str).values,
        da_b["channel"].astype(str).values,
    )

    # event count (assumes both have event dim; your pipeline uses _ensure_dims upstream)
    n_events = int(min(da_a.sizes.get("event", 1), da_b.sizes.get("event", 1)))

    # Track event-level close mismatch across ALL channels
    any_close_mismatch_by_event = np.zeros(n_events, dtype=bool)

    for ch in common_ch:
        da1 = da_a.sel({"channel": ch})
        da2 = da_b.sel({"channel": ch})

        a = np.asarray(da1.data)
        b = np.asarray(da2.data)

        # squeeze channel dim away; expect (event, time) or (time,)
        a = np.squeeze(a)
        b = np.squeeze(b)

        # ensure 2D: (event, time)
        if a.ndim == 1:
            a = a[None, :]
        if b.ndim == 1:
            b = b[None, :]

        # crop to min events and min time
        E = min(a.shape[0], b.shape[0], n_events)
        a = a[:E]
        b = b[:E]

        a2, b2, m = _crop_time_to_min(a, b)  # crops last axis (time)
        # shapes now (E, m)

        both_nan = np.isnan(a2) & np.isnan(b2)
        exact_bad = ~((a2 == b2) | both_nan)
        close_bad = ~np.isclose(a2, b2, rtol=RTOL, atol=ATOL, equal_nan=True)

        # event-level aggregation for THIS channel
        any_exact_this_ch = bool(np.any(exact_bad))
        any_close_this_ch = bool(np.any(close_bad))

        if any_exact_this_ch:
            exact_fail.append(ch)
        if any_close_this_ch:
            close_fail.append(ch)

        # update global event flags: event differs if ANY timepoint differs for this channel
        any_close_mismatch_by_event[:E] |= np.any(close_bad, axis=1)

        # per-event stats for this channel
        diff = a2 - b2
        invalid = both_nan | np.isnan(a2) | np.isnan(b2)
        diff = np.where(invalid, np.nan, diff)
        abs_diff = np.abs(diff)

        n_close_per_event = close_bad.sum(axis=1).astype(int)

        mean_abs = np.nanmean(abs_diff, axis=1)
        max_abs  = np.nanmax(abs_diff, axis=1)
        mean_signed = np.nanmean(diff, axis=1)
        std_diff = np.nanstd(diff, axis=1)
        mse_channel = np.nanmean(diff**2, axis=1)

        for ev_i in range(E):
            rows.append(dict(
                comparison=f"{label_a} vs {label_b}",
                channel=str(ch),
                event=int(ev_i),
                n_close_diff=int(n_close_per_event[ev_i]),
                mean_abs_diff=float(mean_abs[ev_i]) if np.isfinite(mean_abs[ev_i]) else np.nan,
                max_abs_diff=float(max_abs[ev_i]) if np.isfinite(max_abs[ev_i]) else np.nan,
                mean_signed_diff=float(mean_signed[ev_i]) if np.isfinite(mean_signed[ev_i]) else np.nan,
                std_diff=float(std_diff[ev_i]) if np.isfinite(std_diff[ev_i]) else np.nan,
                mse_channel=float(mse_channel[ev_i]) if np.isfinite(mse_channel[ev_i]) else np.nan,
                time_compared_samples=int(m),
            ))

    df = pd.DataFrame(rows)

    close_diff_event_indices = np.where(any_close_mismatch_by_event)[0].tolist()
    return df, exact_fail, close_fail, common_ch, close_diff_event_indices, n_events


# def compare_time_coord_pairs(label_a: str, da_a: xr.DataArray, label_b: str, da_b: xr.DataArray):
#     """
#     Compare the *time coordinate values* between two EEG DataArrays.

#     Assumes:
#       - da_* has a 'time' coordinate or dimension
#       - comparison is by index (crop to min length), not by alignment

#     Returns:
#       df_time_stats : pd.DataFrame with a single row of summary stats
#       exact_fail    : bool (True if any exact mismatch)
#       close_fail    : bool (True if any allclose mismatch)
#       diff_vec      : np.ndarray of time differences (time_a - time_b) on compared indices (NaN where invalid)
#     """
#     if "time" not in da_a.dims or "time" not in da_b.dims:
#         raise ValueError(f"Both inputs must have a 'time' dimension. Got da_a.dims={da_a.dims}, da_b.dims={da_b.dims}")

#     # Pull time coordinate arrays if present; else fall back to index.
#     # (Usually time is a coordinate of the 'time' dim.)
#     if "time" in da_a.coords:
#         t_a = np.asarray(da_a["time"].values)
#     else:
#         t_a = np.arange(da_a.sizes["time"], dtype=float)

#     if "time" in da_b.coords:
#         t_b = np.asarray(da_b["time"].values)
#     else:
#         t_b = np.arange(da_b.sizes["time"], dtype=float)

#     # Ensure 1D
#     t_a = np.squeeze(t_a)
#     t_b = np.squeeze(t_b)

#     # Crop by index to common length
#     t_a2, t_b2, m = _crop_time_to_min(t_a, t_b)

#     # NaN-safe exact + close
#     both_nan = np.isnan(t_a2) & np.isnan(t_b2)
#     exact_bad = ~((t_a2 == t_b2) | both_nan)
#     close_bad = ~np.isclose(t_a2, t_b2, rtol=RTOL, atol=ATOL, equal_nan=True)

#     diff = t_a2 - t_b2
#     invalid = both_nan | np.isnan(t_a2) | np.isnan(t_b2)
#     diff = np.where(invalid, np.nan, diff)
#     abs_diff = np.abs(diff)

#     n_exact = int(np.sum(exact_bad))
#     n_close = int(np.sum(close_bad))

#     mean_abs = float(np.nanmean(abs_diff)) if np.isfinite(abs_diff).any() else np.nan
#     max_abs  = float(np.nanmax(abs_diff))  if np.isfinite(abs_diff).any() else np.nan
#     mean_signed = float(np.nanmean(diff))  if np.isfinite(diff).any() else np.nan
#     std_diff = float(np.nanstd(diff))      if np.isfinite(diff).any() else np.nan
#     mse_time = float(np.nanmean(diff**2))  if np.isfinite(diff).any() else np.nan

#     df = pd.DataFrame([dict(
#         comparison=f"{label_a} vs {label_b}",
#         time_len_a=int(len(t_a)),
#         time_len_b=int(len(t_b)),
#         time_compared_samples=int(m),

#         n_exact_time_diff=n_exact,
#         n_close_time_diff=n_close,

#         mean_abs_time_diff=mean_abs,
#         max_abs_time_diff=max_abs,
#         mean_signed_time_diff=mean_signed,
#         std_time_diff=std_diff,
#         mse_time=mse_time,
#     )])

#     exact_fail = (n_exact != 0)
#     close_fail = (n_close != 0)

#     return df, exact_fail, close_fail, diff

def compare_time_coord_pairs(label_a: str, da_a: xr.DataArray, label_b: str, da_b: xr.DataArray):
    """
    Event-aware comparison of time coordinate values.

    Returns:
      df_time_stats (single row per pair, but includes event mismatch counts/indices),
      exact_fail (bool),
      close_fail (bool),
      diff (np.ndarray of shape (E, T) after crop; NaN where invalid),
      close_diff_event_indices (list[int]),
      n_events (int)
    """
    if "time" not in da_a.dims or "time" not in da_b.dims:
        raise ValueError(f"Both inputs must have a 'time' dimension. Got da_a.dims={da_a.dims}, da_b.dims={da_b.dims}")

    # pull coords (fallback to index)
    t_a = np.asarray(da_a["time"].values) if "time" in da_a.coords else np.arange(da_a.sizes["time"], dtype=float)
    t_b = np.asarray(da_b["time"].values) if "time" in da_b.coords else np.arange(da_b.sizes["time"], dtype=float)

    # normalize to (event, time)
    n_events = int(min(da_a.sizes.get("event", 1), da_b.sizes.get("event", 1)))

    def _to_event_time(t):
        t = np.asarray(t)
        if t.ndim == 1:
            return np.tile(t[None, :], (n_events, 1))
        if t.ndim == 2:
            return t[:n_events]
        raise ValueError(f"Unsupported time coord ndim={t.ndim}; expected 1 or 2.")

    Ta = _to_event_time(t_a)
    Tb = _to_event_time(t_b)

    # crop time to min length
    m = min(Ta.shape[1], Tb.shape[1])
    Ta = Ta[:, :m]
    Tb = Tb[:, :m]

    both_nan = np.isnan(Ta) & np.isnan(Tb)
    exact_bad = ~((Ta == Tb) | both_nan)
    close_bad = ~np.isclose(Ta, Tb, rtol=RTOL, atol=ATOL, equal_nan=True)

    diff = Ta - Tb
    invalid = both_nan | np.isnan(Ta) | np.isnan(Tb)
    diff = np.where(invalid, np.nan, diff)
    abs_diff = np.abs(diff)

    n_exact = int(np.sum(exact_bad))
    n_close = int(np.sum(close_bad))

    # event-level "does this event differ at all?"
    close_bad_by_event = np.any(close_bad, axis=1)
    close_diff_event_indices = np.where(close_bad_by_event)[0].tolist()

    df = pd.DataFrame([dict(
        comparison=f"{label_a} vs {label_b}",
        time_len_a=int(da_a.sizes["time"]),
        time_len_b=int(da_b.sizes["time"]),
        time_compared_samples=int(m),

        n_events=int(n_events),
        n_close_diff_events=int(close_bad_by_event.sum()),
        close_diff_event_indices=close_diff_event_indices,

        n_exact_time_diff=n_exact,
        n_close_time_diff=n_close,

        mean_abs_time_diff=float(np.nanmean(abs_diff)) if np.isfinite(abs_diff).any() else np.nan,
        max_abs_time_diff=float(np.nanmax(abs_diff)) if np.isfinite(abs_diff).any() else np.nan,
        mean_signed_time_diff=float(np.nanmean(diff)) if np.isfinite(diff).any() else np.nan,
        std_time_diff=float(np.nanstd(diff)) if np.isfinite(diff).any() else np.nan,
        mse_time=float(np.nanmean(diff**2)) if np.isfinite(diff).any() else np.nan,
    )])

    exact_fail = (n_exact != 0)
    close_fail = (n_close != 0)

    return df, exact_fail, close_fail, diff, close_diff_event_indices, n_events




def compare_eeg_sources(
    eeg_dict,
    *,
    subject=None,
    experiment=None,
    session=None,
    options=[
        "strip_metadata",
        "print_channel_coord_summary",
        "compare_raw_signals",
        "print_raw_stats",
        "compare_time_coords",
        "print_time_stats",
        "print_mismatched_channels",
        "print_raw_data",
    ],
    channel_dim="channel",
):
    """
    Compare N EEG sources WITHOUT xarray alignment.

    Behavior:
      - Standardizes dim order (event, channel, time)
      - Strips event metadata (optional)
      - Reports channel overlap (pairwise)
      - Pairwise comparisons:
          * channels: intersection by name
          * time: compare by sample index (crop to min length)
    """
    if len(options) <= 0:
        print("no options selected")

    names = list(eeg_dict.keys())
    eegs = list(eeg_dict.values())
    if len(eegs) < 2:
        raise ValueError("Need at least 2 sources to compare.")

    options_set = set(options)

    # ----------------------------
    # Standardize dims (+ optional metadata strip)
    # ----------------------------
    eegs_std = []
    strip_metadata = "strip_metadata" in options_set
    for da in eegs:
        stripped_da = strip_event_metadata(da) if strip_metadata else da
        eegs_std.append(_ensure_dims(stripped_da))

    # ----------------------------
    # Channel summaries (no alignment)
    # ----------------------------
    if "print_channel_coord_summary" in options_set:
        print("\n================ CHANNEL SUMMARY (pre-compare; no alignment) ================")
        for i in range(len(eegs_std)):
            for j in range(i + 1, len(eegs_std)):
                summarize_channel_coords(eegs_std[i], eegs_std[j], channel_dim)

    # Outputs
    df_stats_raw = None
    df_stats_time = None
    df_summary_raw = None  # <-- NEW: DataFrame summary over channels per pair

    # helper for stable nan summaries
    def _nanmean(x):
        x = pd.to_numeric(x, errors="coerce")
        return float(np.nanmean(x)) if np.isfinite(x).any() else np.nan

    def _nanmax(x):
        x = pd.to_numeric(x, errors="coerce")
        return float(np.nanmax(x)) if np.isfinite(x).any() else np.nan

    # ----------------------------
    # Pairwise comparisons: RAW SIGNALS
    # ----------------------------
    if "compare_raw_signals" in options_set:
        print("\n================ PAIRWISE COMPARISONS RAW SIGNAL ================")
        per_channel_frames = []
        summary_rows = []

        for i in range(len(eegs_std)):
            for j in range(i + 1, len(eegs_std)):
                a_name = names[i]
                b_name = names[j]

                # df_pair, exact_fail, close_fail, common_ch = compare_raw_signal_pairs(
                #     a_name, eegs_std[i], b_name, eegs_std[j]
                # )
                df_pair, exact_fail, close_fail, common_ch, close_diff_event_indices, n_events = compare_raw_signal_pairs(
                    a_name, eegs_std[i], b_name, eegs_std[j]
                )

                per_channel_frames.append(df_pair)

                # samples compared = min time length (since compare_raw_signal_pairs crops)
                ta = int(eegs_std[i].sizes.get("time", 0))
                tb = int(eegs_std[j].sizes.get("time", 0))
                n_samples_compared = int(min(ta, tb))
                print(
                    f"{a_name} vs {b_name}: time lengths {ta} vs {tb} "
                    f"(comparing first {n_samples_compared} samples)"
                )

                # Aggregate across channels for this pair
                summary_rows.append(
                    dict(
                        # session identifiers (requested)
                        subject=subject,
                        experiment=experiment,
                        session=session,

                        comparison=f"{a_name} vs {b_name}",
                        source_a=a_name,
                        source_b=b_name,

                        n_common_channels=int(len(common_ch)),
                        n_exact_diff_channels=int(len(exact_fail)),
                        n_close_diff_channels=int(len(close_fail)),

                        # aggregate metrics across channels (means/maxes of per-channel metrics)
                        mean_abs_diff=_nanmean(df_pair["mean_abs_diff"]) if len(df_pair) else np.nan,
                        max_abs_diff=_nanmax(df_pair["max_abs_diff"]) if len(df_pair) else np.nan,
                        mean_signed_diff=_nanmean(df_pair["mean_signed_diff"]) if len(df_pair) else np.nan,
                        std_diff=_nanmean(df_pair["std_diff"]) if len(df_pair) else np.nan,
                        mse=_nanmean(df_pair["mse_channel"]) if len(df_pair) else np.nan,

                        n_samples_compared=n_samples_compared,

                        exact_diff_channels=list(exact_fail),
                        close_diff_channels=list(close_fail),
                        n_events=int(n_events),
                        n_close_diff_events=int(len(close_diff_event_indices)),
                        close_diff_event_indices=list(close_diff_event_indices),

                    )
                )

        df_stats_raw = (
            pd.concat(per_channel_frames, axis=0, ignore_index=True)
            if per_channel_frames else pd.DataFrame()
        )

        # add subject/session/experiment columns to per-channel df too (requested)
        if len(df_stats_raw) > 0:
            df_stats_raw.insert(0, "session", session)
            df_stats_raw.insert(0, "experiment", experiment)
            df_stats_raw.insert(0, "subject", subject)

        df_summary_raw = pd.DataFrame(summary_rows)

        pd.set_option("display.max_rows", None)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 200)

        if "print_raw_stats" in options_set:
            print("\n================ PER-CHANNEL STATS (RAW SIGNAL; ALL PAIRS) ================")
            if len(df_stats_raw) == 0:
                print("[WARN] No raw-signal stats produced (no common channels across any pairs?)")
            else:
                print(df_stats_raw.to_string(index=False))

        if "print_mismatched_channels" in options_set:
            print("\n================ SUMMARY (RAW SIGNAL; aggregated) ================")
            if len(df_summary_raw) == 0:
                print("[WARN] No raw-signal summary produced.")
            else:
                print(df_summary_raw.to_string(index=False))

    # ----------------------------
    # Pairwise comparisons: TIME COORDS
    # ----------------------------
    if "compare_time_coords" in options_set:
        print("\n================ PAIRWISE COMPARISONS TIME COORDS ================")
        time_frames = []

        for i in range(len(eegs_std)):
            for j in range(i + 1, len(eegs_std)):
                a_name = names[i]
                b_name = names[j]
                # df, exact_fail, close_fail, diff, close_diff_event_indices, n_events
                df_time, exact_fail, close_fail, diff_vec, close_diff_event_indices, n_events = compare_time_coord_pairs(
                    a_name, eegs_std[i], b_name, eegs_std[j]
                )
                time_frames.append(df_time)

                ta = int(eegs_std[i].sizes.get("time", 0))
                tb = int(eegs_std[j].sizes.get("time", 0))
                print(
                    f"{a_name} vs {b_name}: time lengths {ta} vs {tb} "
                    f"(comparing first {min(ta, tb)} time coords)"
                )

        df_stats_time = (
            pd.concat(time_frames, axis=0, ignore_index=True)
            if time_frames else pd.DataFrame()
        )

        # add subject/session/experiment columns to time df too (requested)
        if len(df_stats_time) > 0:
            df_stats_time.insert(0, "session", session)
            df_stats_time.insert(0, "experiment", experiment)
            df_stats_time.insert(0, "subject", subject)

        pd.set_option("display.max_rows", None)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 200)

        if "print_time_stats" in options_set:
            print("\n================ TIME COORD STATS (ALL PAIRS) ================")
            if len(df_stats_time) == 0:
                print("[WARN] No time-coord stats produced.")
            else:
                print(df_stats_time.to_string(index=False))

    # ----------------------------
    # Optional raw prints
    # ----------------------------
    if "print_raw_data" in options_set:
        for nm, da in zip(names, eegs_std):
            print_raw_data(nm, da)

    # NOTE: summary_time removed (you said it duplicates df_time)
    return {
        "df_raw": df_stats_raw,
        "df_raw_summary": df_summary_raw,  # <-- DataFrame summary for raw signals
        "df_time": df_stats_time,
        # "eegs_std": eegs_std,
    }
