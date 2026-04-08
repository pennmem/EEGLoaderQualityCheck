"""BIDS data loading helpers — uses CMLBIDSReader for all BIDS access.

CMLBIDSReader API (from bidsreader package):
    reader = CMLBIDSReader(subject, session, task, root)
    reader.load_events(event_type=)     -> pd.DataFrame
    reader.load_raw(acquisition=)       -> mne.io.BaseRaw
    reader.load_epochs(tmin, tmax, events=, acquisition=, ...) -> mne.Epochs
    reader.load_electrodes()            -> pd.DataFrame
    reader.load_channels(acquisition=)  -> pd.DataFrame
    reader.load_combined_channels(acquisition=) -> pd.DataFrame
    reader.is_intracranial()            -> bool
    reader.space                        -> str
    reader.device                       -> str ("eeg" or "ieeg")
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import mne
import numpy as np
import pandas as pd
import xarray as xr
from ptsa.data.timeseries import TimeSeries

import sys
sys.path.append("/home1/zrentala/bidsreader")
from bidsreader import (
    CMLBIDSReader,
    mne_raw_to_ptsa,
    mne_epochs_to_ptsa,
    filter_by_trial_types as _filter_by_trial_types,
    filter_events_df_by_trial_types as _filter_events_df_by_trial_types,
    convert_unit as _convert_unit,
)


# ======================================================================
# Reader factory
# ======================================================================

def get_reader(
    subject: str,
    experiment: str,
    session: Union[str, int],
    bids_root: str,
) -> CMLBIDSReader:
    """Create a CMLBIDSReader for the given session."""
    reader = CMLBIDSReader(
        subject=subject,
        session=session,
        task=experiment,
        root=bids_root,
    )
    if not reader.is_intracranial():
        reader.set_fields(task=experiment.lower())
    return reader


# ======================================================================
# Events
# ======================================================================

def load_bids_events(
    reader: CMLBIDSReader,
    *,
    event_type: str = "beh",
) -> pd.DataFrame:
    """Load BIDS events via CMLBIDSReader.

    Parameters
    ----------
    reader : CMLBIDSReader
        Pre-configured reader.
    event_type : str
        "beh" for behavioral events, or the device ("eeg"/"ieeg") for
        EEG-aligned events.
    """
    return reader.load_events(event_type=event_type)


# ======================================================================
# Raw EEG
# ======================================================================

def load_bids_raw(
    reader: CMLBIDSReader,
    *,
    acquisition: Optional[str] = None,
) -> mne.io.BaseRaw:
    """Load a BIDS raw object via CMLBIDSReader."""
    return reader.load_raw(acquisition=acquisition)


def raw_to_xarray(
    raw: mne.io.BaseRaw,
    *,
    convert_ms: bool = True,
) -> xr.DataArray:
    """Convert an MNE Raw to a 3-D xarray (event=1, channel, time)."""
    scale = 1000.0 if convert_ms else 1.0
    return xr.DataArray(
        raw.get_data()[None, :, :],
        dims=("event", "channel", "time"),
        coords={
            "event": [0],
            "channel": raw.ch_names,
            "time": raw.times * scale,
            "samplerate": raw.info["sfreq"],
        },
        name="eeg",
    )


def raw_to_ptsa(
    raw: mne.io.BaseRaw,
    **kwargs,
):
    """Convert MNE Raw to PTSA TimeSeries."""
    return mne_raw_to_ptsa(raw, **kwargs)


# ======================================================================
# Epoched EEG
# ======================================================================

def load_bids_epochs(
    reader: CMLBIDSReader,
    tmin: float,
    tmax: float,
    *,
    events: Optional[pd.DataFrame] = None,
    acquisition: Optional[str] = None,
    baseline=None,
    preload: bool = True,
    channels: Optional[list] = None,
) -> mne.Epochs:
    """Load BIDS epochs via CMLBIDSReader.load_epochs."""
    return reader.load_epochs(
        tmin=tmin,
        tmax=tmax,
        events=events,
        acquisition=acquisition,
        baseline=baseline,
        preload=preload,
        channels=channels,
    )


def epochs_to_ptsa(
    epochs: mne.Epochs,
    events: pd.DataFrame,
):
    """Convert MNE Epochs to PTSA TimeSeries."""
    return mne_epochs_to_ptsa(epochs, events)


# ======================================================================
# Electrodes / channels
# ======================================================================

def load_bids_electrodes(reader: CMLBIDSReader) -> pd.DataFrame:
    """Load electrodes.tsv via CMLBIDSReader."""
    return reader.load_electrodes()


def load_bids_channels(
    reader: CMLBIDSReader,
    acquisition: Optional[str] = None,
) -> pd.DataFrame:
    """Load channels.tsv via CMLBIDSReader."""
    return reader.load_channels(acquisition=acquisition)


def load_bids_combined_channels(
    reader: CMLBIDSReader,
    acquisition: Optional[str] = None,
) -> pd.DataFrame:
    """Load combined channels + electrodes via CMLBIDSReader."""
    return reader.load_combined_channels(acquisition=acquisition)


# ======================================================================
# Filtering helpers (delegate to bidsreader standalone functions)
# ======================================================================

def filter_by_trial_types(trial_types, **kwargs):
    """Delegate to bidsreader.filter_by_trial_types."""
    return _filter_by_trial_types(trial_types, **kwargs)


def filter_events_df(events_df: pd.DataFrame, trial_types):
    """Delegate to bidsreader.filter_events_df_by_trial_types."""
    return _filter_events_df_by_trial_types(events_df, trial_types)

def convert_unit(data, target, *, current_unit=None, copy=True):
    return _convert_unit(data=data, target=target, current_unit=current_unit, copy=copy)
