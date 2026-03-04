"""BIDS data loading helpers — uses BIDSReader for all BIDS access.

BIDSReader API (from bidsreader package):
    reader = BIDSReader(subject, session, task, root)
    reader.load_events(event_type=)     -> pd.DataFrame
    reader.load_raw(acquisition=)       -> mne.io.BaseRaw
    reader.load_epochs(tmin, tmax, events=, acquisition=, ...) -> mne.Epochs
    reader.load_electrodes()            -> pd.DataFrame
    reader.load_channels(acquisition=)  -> pd.DataFrame
    reader.load_combined_channels(acquisition=) -> pd.DataFrame
    reader.is_intracranial()            -> bool
    reader.space                        -> str
    BIDSReader.mne_epochs_to_ptsa(epochs, events) -> TimeSeries
    BIDSReader.mne_raw_to_ptsa(raw)     -> TimeSeries
    BIDSReader.filter_by_trial_types(...)
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
from bidsreader import BIDSReader


# ======================================================================
# Reader factory
# ======================================================================

def get_reader(
    subject: str,
    experiment: str,
    session: Union[str, int],
    bids_root: str,
) -> BIDSReader:
    """Create a BIDSReader for the given session."""
    reader = BIDSReader(
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
    reader: BIDSReader,
    *,
    event_type: str = "beh",
) -> pd.DataFrame:
    """Load BIDS events via BIDSReader.

    Parameters
    ----------
    reader : BIDSReader
        Pre-configured reader.
    event_type : str
        "beh" for behavioral events, or the eeg_type ("eeg"/"ieeg") for
        EEG-aligned events.
    """
    return reader.load_events(event_type=event_type)


# ======================================================================
# Raw EEG
# ======================================================================

def load_bids_raw(
    reader: BIDSReader,
    *,
    acquisition: Optional[str] = None,
) -> mne.io.BaseRaw:
    """Load a BIDS raw object via BIDSReader."""
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
    """Convert MNE Raw to PTSA TimeSeries via BIDSReader static method."""
    return BIDSReader.mne_raw_to_ptsa(raw, **kwargs)


# ======================================================================
# Epoched EEG
# ======================================================================

def load_bids_epochs(
    reader: BIDSReader,
    tmin: float,
    tmax: float,
    *,
    events: Optional[pd.DataFrame] = None,
    acquisition: Optional[str] = None,
    baseline=None,
    preload: bool = True,
    channels: Optional[list] = None,
) -> mne.Epochs:
    """Load BIDS epochs via BIDSReader.load_epochs."""
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
    """Convert MNE Epochs to PTSA TimeSeries via BIDSReader static method."""
    return BIDSReader.mne_epochs_to_ptsa(epochs, events)


# ======================================================================
# Electrodes / channels
# ======================================================================

def load_bids_electrodes(reader: BIDSReader) -> pd.DataFrame:
    """Load electrodes.tsv via BIDSReader."""
    return reader.load_electrodes()


def load_bids_channels(
    reader: BIDSReader,
    acquisition: Optional[str] = None,
) -> pd.DataFrame:
    """Load channels.tsv via BIDSReader."""
    return reader.load_channels(acquisition=acquisition)


def load_bids_combined_channels(
    reader: BIDSReader,
    acquisition: Optional[str] = None,
) -> pd.DataFrame:
    """Load combined channels + electrodes via BIDSReader."""
    return reader.load_combined_channels(acquisition=acquisition)


# ======================================================================
# Filtering helpers (delegate to BIDSReader static methods)
# ======================================================================

def filter_by_trial_types(trial_types, **kwargs):
    """Delegate to BIDSReader.filter_by_trial_types."""
    return BIDSReader.filter_by_trial_types(trial_types, **kwargs)


def filter_events_df(events_df: pd.DataFrame, trial_types):
    """Delegate to BIDSReader.filter_events_df_by_trial_types."""
    return BIDSReader.filter_events_df_by_trial_types(events_df, trial_types)

def convert_unit(data, target, *, current_unit=None, copy=True):
    return BIDSReader.convert_unit(data=data, target=target, current_unit=current_unit, copy=copy)