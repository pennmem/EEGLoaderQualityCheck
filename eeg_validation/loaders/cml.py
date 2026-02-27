"""CML data loading helpers."""

from __future__ import annotations

from typing import Optional, Union

import cmlreaders as cml
import pandas as pd


def load_cml_events(
    subject: str,
    experiment: str,
    session: int,
    *,
    localization: Optional[int] = None,
    montage: Optional[int] = None,
) -> pd.DataFrame:
    reader = cml.CMLReader(
        subject=subject,
        experiment=experiment,
        session=session,
        localization=localization,
        montage=montage,
    )
    return reader.load("events")


def load_cml_eeg_raw(
    subject: str,
    experiment: str,
    session: int,
    *,
    localization: Optional[int] = None,
    montage: Optional[int] = None,
    scheme: Optional[pd.DataFrame] = None,
):
    """Load continuous EEG as PTSA TimeSeries."""
    reader = cml.CMLReader(
        subject=subject,
        experiment=experiment,
        session=session,
        localization=localization,
        montage=montage,
    )
    return reader.load_eeg(scheme=scheme).to_ptsa()


def load_cml_eeg_epoched(
    subject: str,
    experiment: str,
    session: int,
    events: pd.DataFrame,
    rel_start: float,
    rel_stop: float,
    *,
    localization: Optional[int] = None,
    montage: Optional[int] = None,
    scheme: Optional[pd.DataFrame] = None,
):
    """Load epoched EEG as PTSA TimeSeries."""
    reader = cml.CMLReader(
        subject=subject,
        experiment=experiment,
        session=session,
        localization=localization,
        montage=montage,
    )
    return reader.load_eeg(events, scheme=scheme, rel_start=rel_start, rel_stop=rel_stop).to_ptsa()


def load_cml_contacts_and_pairs(
    subject: str,
    experiment: str,
    session: int,
    localization: int,
    montage: int,
):
    """Return (contacts_df, pairs_df)."""
    reader = cml.CMLReader(
        subject=subject,
        experiment=experiment,
        session=session,
        localization=localization,
        montage=montage,
    )
    return reader.load("contacts"), reader.load("pairs")
