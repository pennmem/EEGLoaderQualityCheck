"""Load EDF files into xarray DataArrays via pyedflib."""

from __future__ import annotations

import numpy as np
import pyedflib
import xarray as xr


_UNIT_SCALE = {
    "uv": 1e-6,
    "µv": 1e-6,
    "mv": 1e-3,
    "v": 1.0,
}


def load_edf_as_xarray(path: str, *, event_dim_name: str = "event") -> xr.DataArray:
    """Read an EDF file and return an xarray with shape (1, channels, time) in volts."""
    f = pyedflib.EdfReader(path)
    try:
        n_channels = f.signals_in_file
        ch_names = list(f.getSignalLabels())
        sfreq = float(f.getSampleFrequency(0))
        n_samples = int(f.getNSamples()[0])

        data_V = []
        for ch in range(n_channels):
            x = f.readSignal(ch)
            unit = f.getPhysicalDimension(ch).strip().lower()
            scale = _UNIT_SCALE.get(unit, None)
            if scale is None:
                print(
                    f"Unknown physical unit '{unit}' for channel {ch} ({ch_names[ch]}). "
                    "Assuming volts."
                )
                scale = 1.0
            data_V.append(x * scale)

        data = np.vstack(data_V)[None, :, :]  # (1, ch, time)
        times = np.arange(n_samples) / sfreq

        return xr.DataArray(
            data,
            dims=(event_dim_name, "channel", "time"),
            coords={
                event_dim_name: [0],
                "channel": ch_names,
                "time": times,
                "samplerate": sfreq,
            },
            name="eeg",
            attrs={"units": "V", "source": "pyedflib"},
        )
    finally:
        f.close()
