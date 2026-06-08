"""DigitalSignalComparator — compare raw integer samples in EDF vs BDF.

This bypasses every gain/offset conversion in EDF/BDF and compares the raw
digitized values directly. The only "real" question for two readers of the
same recording is whether the underlying integer samples round-trip
losslessly through the BIDS re-encoding; this comparator answers it.

A 16-bit EDF and a 24-bit BDF derived from the same recording should agree
after a left-shift by 8 (i.e. ``bdf_int == edf_int * 256``). Differences of
0 mean lossless round-trip; differences of ≤1 LSB mean unavoidable
rounding from a non-power-of-2 rescale; differences > 1 LSB mean a real
encoder bug.

The summary frame is a single row whose column names mirror
``SignalComparator`` (no ``int`` infix — the comparator only ever sees
ints) so the same notebook plotting helper works on both.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .base import Comparator, ComparisonResult
from .utils import crop_to_min_length


def _read_edf_bdf_digital(path: str):
    """Read an EDF or BDF file and return ``(format, labels, [int64 arrays])``.

    Pure-numpy reader — no pyedflib dependency. Detects BDF (24-bit) by the
    leading 0xFF byte, otherwise treats the file as EDF (16-bit). Handles
    sign-extension for BDF int24 samples.

    The first element of the return tuple is the string ``"EDF"`` or
    ``"BDF"`` so callers can pick a sample-scale factor automatically.
    """
    with open(path, "rb") as f:
        h1 = f.read(256)
        is_bdf = h1[0] == 0xFF
        sample_size = 3 if is_bdf else 2
        n_records = int(h1[236:244])
        n_signals = int(h1[252:256])
        h2 = f.read(n_signals * 256)
        data = f.read()

    labels = [
        h2[i * 16 : (i + 1) * 16].decode("ascii", errors="replace").strip()
        for i in range(n_signals)
    ]
    # Per-signal "samples_per_record" lives at byte offset 216*ns within h2.
    spr_off = (16 + 80 + 8 + 8 + 8 + 8 + 8 + 80) * n_signals
    samples_per_record = [
        int(h2[spr_off + i * 8 : spr_off + (i + 1) * 8]) for i in range(n_signals)
    ]
    record_size = sum(samples_per_record) * sample_size

    expected = n_records * record_size
    if len(data) < expected:
        raise ValueError(
            f"{path}: truncated data block ({len(data)} < {expected})"
        )
    arr = np.frombuffer(data[:expected], dtype=np.uint8).reshape(
        n_records, record_size
    )

    # Per-signal byte offsets within each record.
    offsets = [0]
    for s in samples_per_record:
        offsets.append(offsets[-1] + s * sample_size)

    signals: List[np.ndarray] = []
    for i in range(n_signals):
        seg = arr[:, offsets[i] : offsets[i + 1]]  # (n_records, spr[i]*sample_size)
        if is_bdf:
            seg3 = seg.reshape(-1, 3)
            ints = (
                seg3[:, 0].astype(np.int32)
                | (seg3[:, 1].astype(np.int32) << 8)
                | (seg3[:, 2].astype(np.int32) << 16)
            )
            # Sign-extend from bit 23.
            ints = np.where(ints & 0x800000, ints - 0x1000000, ints).astype(np.int64)
        else:
            ints = np.frombuffer(seg.tobytes(), dtype="<i2").astype(np.int64)
        signals.append(ints)
    fmt = "BDF" if is_bdf else "EDF"
    return fmt, labels, signals


# Native digital range per format. Used to derive the auto-detected
# A→B scale factor: ratio of B's range to A's range, rounded to the
# nearest power of 2.
_FORMAT_DIGITAL_RANGE = {
    "EDF": 32768,    # int16: ±32768 (2^15)
    "BDF": 8388608,  # int24: ±8388608 (2^23)
}


def _auto_int_scale(fmt_a: str, fmt_b: str) -> Tuple[int, int]:
    """Pick lossless ``(scale_a, scale_b)`` to bring two formats to a common range.

    The wider-range side keeps scale=1; the narrower-range side gets
    upscaled by ``range_wider // range_narrower`` so the comparison
    happens in the wider int domain with no precision loss.

    Examples (all lossless):
      - EDF↔EDF → (1, 1)
      - BDF↔BDF → (1, 1)
      - EDF↔BDF → (256, 1)   (a is upscaled into BDF range)
      - BDF↔EDF → (1, 256)   (b is upscaled into BDF range)
    """
    if fmt_a not in _FORMAT_DIGITAL_RANGE or fmt_b not in _FORMAT_DIGITAL_RANGE:
        raise ValueError(f"unknown formats: a={fmt_a!r} b={fmt_b!r}")
    range_a = _FORMAT_DIGITAL_RANGE[fmt_a]
    range_b = _FORMAT_DIGITAL_RANGE[fmt_b]
    if range_a >= range_b:
        ratio = range_a // range_b
        if ratio * range_b != range_a:
            raise ValueError(
                f"cannot auto-scale {fmt_a}↔{fmt_b}: ratio {range_a}/{range_b} "
                "is not an integer"
            )
        return 1, int(ratio)
    ratio = range_b // range_a
    if ratio * range_a != range_b:
        raise ValueError(
            f"cannot auto-scale {fmt_a}↔{fmt_b}: ratio {range_b}/{range_a} "
            "is not an integer"
        )
    return int(ratio), 1


class DigitalSignalComparator(Comparator):
    """Compare integer sample streams in two EDF/BDF files.

    Parameters
    ----------
    int_scale_a_to_b : int or None
        Multiplier applied to A's int samples before subtracting from B.
        ``None`` (default) auto-detects the formats of both files and
        picks the lossless rescale factor: 1 for same-format pairs, 256
        for EDF→BDF. Pass an explicit integer to override.
    """

    def __init__(self, *, int_scale_a_to_b: Optional[int] = None):
        self.int_scale_a_to_b = (
            None if int_scale_a_to_b is None else int(int_scale_a_to_b)
        )

    # ------------------------------------------------------------------
    def compare(
        self,
        source_a: Union[str, tuple],
        source_b: Union[str, tuple],
        *,
        label_a: str = "EDF",
        label_b: str = "BDF",
        subject: Optional[str] = None,
        experiment: Optional[str] = None,
        session: Optional[Union[str, int]] = None,
        acquisition: Optional[str] = None,
    ) -> ComparisonResult:
        """Compare integer samples from two sources.

        Each source can be:
        - A file path (str) — read via ``_read_edf_bdf_digital``.
        - A ``(fmt, labels, signals)`` tuple — pre-loaded data where
          ``fmt`` is ``"EDF"`` or ``"BDF"``, ``labels`` is a list of
          channel names, and ``signals`` is a list of int64 arrays.
        """
        if isinstance(source_a, str):
            path_a = source_a
            fmt_a, ch_a, sig_a = _read_edf_bdf_digital(source_a)
        else:
            path_a = None
            fmt_a, ch_a, sig_a = source_a
        if isinstance(source_b, str):
            path_b = source_b
            fmt_b, ch_b, sig_b = _read_edf_bdf_digital(source_b)
        else:
            path_b = None
            fmt_b, ch_b, sig_b = source_b
        if self.int_scale_a_to_b is None:
            scale_a, scale_b = _auto_int_scale(fmt_a, fmt_b)
        else:
            # Manual override: legacy single-int means "multiply a by this,
            # leave b alone".
            scale_a, scale_b = self.int_scale_a_to_b, 1
        common = [c for c in ch_a if c in set(ch_b)]
        idx_a = {c: i for i, c in enumerate(ch_a)}
        idx_b = {c: i for i, c in enumerate(ch_b)}

        # Per-channel aggregates aligned with `common`.
        channel_means: List[float] = []
        channel_stds: List[float] = []
        channel_max_abs_diff: List[int] = []
        channel_mean_abs_diff: List[float] = []
        channel_n_diff: List[int] = []
        channel_n_diff_gt_1: List[int] = []
        channel_n_samples: List[int] = []
        channel_mse: List[float] = []

        # Cross-channel diff accumulators (for global scalars).
        global_abs_diff_sum = 0.0
        global_sq_diff_sum = 0.0
        global_signed_diff_sum = 0.0
        global_n = 0
        global_n_diff = 0
        global_n_diff_gt_1 = 0
        global_max_abs_diff = 0

        for ch in common:
            a_int = sig_a[idx_a[ch]]
            b_int = sig_b[idx_b[ch]]

            a_scaled = a_int * scale_a
            b_scaled = b_int * scale_b
            a_scaled, b_scaled, m = crop_to_min_length(a_scaled, b_scaled)

            diff = b_scaled - a_scaled
            abs_diff = np.abs(diff)

            ch_n_diff = int(np.count_nonzero(diff))
            ch_max_abs = int(abs_diff.max()) if m > 0 else 0
            ch_mean_abs = float(abs_diff.mean()) if m > 0 else float("nan")
            ch_n_gt_1 = int(np.count_nonzero(abs_diff > 1))
            ch_mse = float((diff.astype(np.float64) ** 2).mean()) if m > 0 else float("nan")

            channel_means.append(float(b_int.mean()) if m > 0 else float("nan"))
            channel_stds.append(float(b_int.std()) if m > 0 else float("nan"))
            channel_max_abs_diff.append(ch_max_abs)
            channel_mean_abs_diff.append(ch_mean_abs)
            channel_n_diff.append(ch_n_diff)
            channel_n_diff_gt_1.append(ch_n_gt_1)
            channel_n_samples.append(int(m))
            channel_mse.append(ch_mse)

            global_abs_diff_sum += float(abs_diff.sum())
            global_sq_diff_sum += float((diff.astype(np.float64) ** 2).sum())
            global_signed_diff_sum += float(diff.sum())
            global_n += int(m)
            global_n_diff += ch_n_diff
            global_n_diff_gt_1 += ch_n_gt_1
            if ch_max_abs > global_max_abs_diff:
                global_max_abs_diff = ch_max_abs

        # Scalar metrics. Names mirror SignalComparator so the same notebook
        # plot calls work on either summary frame. Units are LSB (digital
        # units), not µV.
        if global_n > 0:
            mean_abs_diff = global_abs_diff_sum / global_n
            mse_scalar = global_sq_diff_sum / global_n
            mean_signed_diff = global_signed_diff_sum / global_n
            std_diff = float(np.sqrt(max(mse_scalar - mean_signed_diff ** 2, 0.0)))
            frac_diff_gt_1 = (global_n_diff_gt_1 / global_n_diff) if global_n_diff else 0.0
        else:
            mean_abs_diff = float("nan")
            mse_scalar = float("nan")
            mean_signed_diff = float("nan")
            std_diff = float("nan")
            frac_diff_gt_1 = float("nan")

        # Map acquisition tag the same way SignalComparator does.
        if acquisition is None or str(acquisition).lower() in ("eeg", "scalp"):
            acq_tag = "scalp"
        else:
            acq_tag = acquisition

        df_summary = pd.DataFrame([{
            "subject": subject,
            "experiment": experiment,
            "session": session,
            "comparison": f"{label_a} vs {label_b}",
            "source_a": label_a,
            "source_b": label_b,
            "acquisition": acq_tag,
            "n_common_channels": len(common),
            "n_samples_per_channel": int(channel_n_samples[0]) if channel_n_samples else 0,
            "mean_abs_diff": float(mean_abs_diff),
            "max_abs_diff": int(global_max_abs_diff),
            "mean_signed_diff": float(mean_signed_diff),
            "std_diff": float(std_diff),
            "mse": float(mse_scalar),
            "n_diff_total": int(global_n_diff),
            "n_diff_gt_1_total": int(global_n_diff_gt_1),
            "frac_diff_gt_1": float(frac_diff_gt_1),
            "common_channels": [str(c) for c in common],
            "channel_means": channel_means,
            "channel_stds": channel_stds,
            "channel_max_abs_diff": channel_max_abs_diff,
            "channel_mean_abs_diff": channel_mean_abs_diff,
            "channel_n_diff": channel_n_diff,
            "channel_n_diff_gt_1": channel_n_diff_gt_1,
            "channel_n_samples": channel_n_samples,
            "channel_mse": channel_mse,
            "format_a": fmt_a,
            "format_b": fmt_b,
            "scale_a": scale_a,
            "scale_b": scale_b,
            "int_scale_auto": self.int_scale_a_to_b is None,
            "path_a": path_a,
            "path_b": path_b,
        }])

        return ComparisonResult(
            ok=(global_n_diff_gt_1 == 0),
            df_summary=df_summary,
            df_detail=pd.DataFrame(),
            df_mismatches=pd.DataFrame(),
            subject=subject,
            experiment=experiment,
            session=session,
            extras={"df_digital_summary": df_summary},
        )
