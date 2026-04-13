"""Digital reader counterpart to bids-convert's edf_digital_writer.

Uses the writer's `resolve_edf_units` cascade verbatim (priority:
source EDF header → CSV → derived) to obtain per-channel
pmin/pmax/dmin/dmax/dim, then inverts via the standard EDF formula:

    gain   = (pmax - pmin) / (dmax - dmin)
    offset = pmin - dmin * gain
    volts  = (data_int * gain + offset) * dim_to_si_scale(dim)

This is the read-side mirror of how the BIDS file was written, so a
round-trip through it lets the comparison pipeline check the writer
against itself instead of against a CSV-based scalar divisor.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Optional

import numpy as np

# bids-convert is a sibling repo on the same machine. Import the writer's
# resolve_edf_units verbatim so the cascade stays in one place — same
# pattern eeg_validation already uses for `bidsreader` (see
# loaders/bids.py).
sys.path.append("/home1/zrentala/bids-convert")
from intracranial.edf_digital_writer import resolve_edf_units  # noqa: E402


_DIM_TO_SI = {"v": 1.0, "mv": 1e-3, "uv": 1e-6, "µv": 1e-6, "nv": 1e-9}


def dim_to_si_scale(dim: str) -> float:
    """Multiplier that converts ``dim`` to Volts. Raises on unknown."""
    if dim is None:
        raise ValueError("dim is None")
    key = dim.strip().lower()
    if key not in _DIM_TO_SI:
        raise ValueError(f"unknown physical_dimension {dim!r}")
    return _DIM_TO_SI[key]


def compute_gain_offset(pmin: float, pmax: float, dmin: int, dmax: int):
    """Standard EDF/BDF reconstruction: ``physical = digital*gain + offset``."""
    if dmax == dmin:
        raise ValueError(f"digital_max == digital_min ({dmin}); cannot compute gain")
    gain = (float(pmax) - float(pmin)) / (float(dmax) - float(dmin))
    offset = float(pmin) - float(dmin) * gain
    return gain, offset


def _source_edf_path(subject: str, experiment: str, session) -> Optional[str]:
    """Mirror writer's `_source_recording_path`. Returns the original
    EDF under ``/protocols/r1/.../current_source/raw_eeg/``, or None.
    """
    index_path = os.path.join(
        "/protocols/r1/subjects", subject,
        "experiments", experiment,
        "sessions", str(session),
        "ephys", "current_source", "index.json",
    )
    if not os.path.exists(index_path):
        return None
    try:
        with open(index_path) as f:
            index = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    files = (index.get("raw_eeg") or {}).get("files") or []
    if not files:
        return None
    candidate = os.path.join(os.path.dirname(index_path), files[0])
    return candidate if os.path.exists(candidate) else None


def cml_to_volts(
    eeg_cml,
    *,
    subject: str,
    experiment: str,
    session,
    conversion_to_V,
    container: str = "EDF",
    channel_dim: str = "channel",
):
    """Convert a CML PTSA / xarray of integer LSBs to Volts.

    Resolves per-channel units via the writer's :func:`resolve_edf_units`
    cascade and applies ``(int * gain + offset) * unit_scale`` per
    channel. Returns a DataArray with the same dims/coords/name and
    a ``units_status`` attr indicating which cascade branch was used.
    Raises ``KeyError`` if any CML channel label is missing from the
    resolved units.
    """
    labels = [str(c) for c in eeg_cml[channel_dim].values]
    data_int = np.asarray(eeg_cml.values)

    signal_units, status = resolve_edf_units(
        labels,
        source_edf_path=_source_edf_path(subject, experiment, session),
        conversion_to_V=float(conversion_to_V) if conversion_to_V else None,
        container=container,
        # Pass un-converted CML samples so the derived branch sees the
        # actual range — same thing the writer does with the int it's
        # about to write.
        data_for_fallback=data_int,
    )

    # Per-channel gain/offset, baked together with the SI scale so a
    # single mul + add gets us straight to Volts.
    gains = np.empty(len(labels), dtype=np.float64)
    offsets = np.empty(len(labels), dtype=np.float64)
    for i, lbl in enumerate(labels):
        if lbl not in signal_units:
            raise KeyError(f"resolve_edf_units returned no entry for {lbl!r}")
        pmin, pmax, dmin, dmax, dim = signal_units[lbl]
        g, o = compute_gain_offset(pmin, pmax, dmin, dmax)
        scale = dim_to_si_scale(dim)
        gains[i] = g * scale
        offsets[i] = o * scale

    # Broadcast over the channel axis regardless of dim ordering.
    ax = eeg_cml.dims.index(channel_dim)
    shape = [1] * eeg_cml.ndim
    shape[ax] = len(labels)
    g_arr = gains.reshape(shape)
    o_arr = offsets.reshape(shape)

    volts = data_int.astype(np.float64) * g_arr + o_arr
    out = eeg_cml.copy(data=volts)
    out.attrs["units_status"] = status
    return out
