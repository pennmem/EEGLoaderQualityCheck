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


def _strip_yc_unhashable_cols(events: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """YC events carry list-of-dict columns (`path`, and `stim_params` for
    YC2) that cmlreaders' to_ptsa() can't factorize into a MultiIndex —
    its `tuple(list)` coercion leaves the inner dicts unhashable. Drop
    them; the comparison pipeline only uses data/channel/time."""
    if events is None:
        return events
    drop = []
    for col in events.columns:
        v = events[col].dropna()
        if not len(v):
            continue
        x = v.iloc[0]
        if isinstance(x, dict) or (isinstance(x, list) and x and isinstance(x[0], dict)):
            drop.append(col)
    return events.drop(columns=drop) if drop else events


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
    container = reader.load_eeg(scheme=scheme)
    if experiment.startswith("YC"):
        container.events = _strip_yc_unhashable_cols(container.events)
    return container.to_ptsa()


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
    if experiment.startswith("YC"):
        events = _strip_yc_unhashable_cols(events)
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
