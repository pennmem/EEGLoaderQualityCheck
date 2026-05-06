"""Attach pair-level region labels from the upstream CML pairs.json.

The neurorad pipeline writes per-pair region labels by running an
independent atlas lookup at each pair's midpoint voxel (see
``event_creation/submission/neurorad_tasks.py`` ``CreateMontageTask``
lines 257-288 and ``ATLAS_NAMES_TABLE`` at 200-205). Those labels can't
be reproduced from contact-level BIDS data, so ``prep_pairs`` in
:mod:`.montage` intentionally omits them. This module reads them
directly from the upstream artifact via ``cmlreaders.CMLReader.load
("pairs")`` and joins onto a BIDS-derived pairs DataFrame by pair label.

Design notes
------------
* Read-only. Does not mutate its input.
* ``cmlreaders`` is a soft dependency — imported at call time and
  raising a clear error if missing, so users without a CML stack can
  still use :mod:`.montage`.
* Matches on pair label (``"LAF1-LAF2"``), which is the only key
  shared between BIDS and the CML localization (CML contact integer
  IDs aren't derivable from BIDS).
"""

from __future__ import annotations

from typing import Iterable, Optional

import pandas as pd


# Region columns we pull through from CML pairs if present. Matches
# FIELD_NAMES_TABLE keys in event_creation/submission/neurorad_tasks.py
# (lines 190-198); each corresponds to one entry in ATLAS_NAMES_TABLE.
_DEFAULT_REGION_COLS: tuple[str, ...] = (
    "ind.region",            # dk (Desikan-Killiany) atlas in fsnative
    "ind.corrected.region",  # dk after brainshift correction
    "avg.region",            # fsaverage (dk projected)
    "avg.corrected.region",
    "mni.region",            # whole_brain atlas
    "hcp.region",            # HCP parcellation
    "stein.region",          # manual atlas (legacy stein)
    "das.region",            # manual atlas (legacy das)
)


def enrich_pairs_with_cml_regions(
    bids_pairs: pd.DataFrame,
    *,
    subject: Optional[str] = None,
    experiment: Optional[str] = None,
    session: Optional[int] = None,
    localization: Optional[int] = 0,
    montage: Optional[int] = 0,
    reader=None,
    label_col: str = "label",
    region_cols: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Return a copy of ``bids_pairs`` with CML region columns joined on.

    Parameters
    ----------
    bids_pairs :
        Pairs DataFrame produced by :func:`.montage.prep_pairs`. Must
        contain a column named ``label_col`` (default ``"label"``) with
        values like ``"LAF1-LAF2"``.
    subject, experiment, session, localization, montage :
        Identifiers passed to :class:`cmlreaders.CMLReader` if
        ``reader`` is not supplied.
    reader :
        Optional ready-made :class:`cmlreaders.CMLReader`. If given,
        the subject/experiment/etc. args are ignored.
    label_col :
        Column name on ``bids_pairs`` that carries the pair label.
    region_cols :
        Iterable of region-column names to carry through from
        ``pairs.json``. Defaults to every ``*.region`` field neurorad
        knows how to emit.

    Returns
    -------
    pandas.DataFrame
        Copy of ``bids_pairs`` with one extra column per entry in
        ``region_cols`` that is actually present in the upstream
        pairs.json. Missing-for-this-subject columns are silently
        skipped (no synthetic NaN columns).
    """

    try:
        from cmlreaders import CMLReader  # soft dep
    except ImportError as exc:
        raise ImportError(
            "enrich_pairs_with_cml_regions requires cmlreaders. Install "
            "it (pip install cmlreaders) or load pairs.json manually."
        ) from exc

    if reader is None:
        if subject is None or experiment is None or session is None:
            raise ValueError(
                "Provide either a ready CMLReader or "
                "(subject, experiment, session)."
            )
        reader = CMLReader(
            subject=subject,
            experiment=experiment,
            session=session,
            localization=localization,
            montage=montage,
        )

    cml_pairs = reader.load("pairs")

    wanted = tuple(region_cols) if region_cols is not None else _DEFAULT_REGION_COLS
    available = [c for c in wanted if c in cml_pairs.columns]

    if "label" not in cml_pairs.columns:
        raise KeyError(
            "cml_pairs is missing the 'label' column — can't join on pair "
            "name. Got columns: %r" % (list(cml_pairs.columns),)
        )

    right = cml_pairs[["label", *available]].copy()
    right["label"] = right["label"].astype("string").str.strip()
    if label_col != "label":
        right = right.rename(columns={"label": label_col})

    out = bids_pairs.copy()
    out[label_col] = out[label_col].astype("string").str.strip()

    return out.merge(right, how="left", on=label_col)
