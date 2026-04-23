"""Prepare BIDS electrodes/channels into CML-compatible contacts and pairs DataFrames."""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


_TYPE_MAP = {"grid": "G", "depth": "D", "strip": "S", "gird": "G"}

_GROUP_COL_CANDIDATES = ("type", "group", "Group", "electrode_type", "contact_type")

# CML atlas/space key -> BIDS-standard space label. Mirrors
# bids-convert/intracranial_BIDS_converter.py CML_TO_BIDS_SPACE; kept
# local so eeg-validation doesn't import from bids-convert.
# Source of truth: pennmem/neurorad_pipeline RELEASE_NOTES.md.
CML_TO_BIDS_SPACE = {
    "mni": "MNI152NLin6ASym",
    "tal": "Talairach",
    "avg": "fsaverage",
    "avg.corrected": "fsaverageBrainshift",
    "ind": "fsnative",
    "ind.corrected": "fsnativeBrainshift",
    "ind.dural": "fsnativeDural",
    "vox": "Pixels",
    "t1_mri": "t1MRI",
}
BIDS_TO_CML_SPACE = {v: k for k, v in CML_TO_BIDS_SPACE.items()}


def _find_group_col(df: pd.DataFrame) -> Optional[str]:
    for cand in _GROUP_COL_CANDIDATES:
        if cand in df.columns:
            return cand
    return None


def _norm_group(g, type_map=_TYPE_MAP):
    if pd.isna(g):
        return pd.NA
    return type_map.get(str(g).strip().lower().replace("gird", "grid"), pd.NA)


def _assign_coords(df: pd.DataFrame, cml_key: str) -> pd.DataFrame:
    """Rename BIDS electrodes.tsv x/y/z to CML-style {cml_key}.x/y/z."""
    return df.rename(columns={
        "x": f"{cml_key}.x",
        "y": f"{cml_key}.y",
        "z": f"{cml_key}.z",
    })


def _dedupe_electrodes(elec: pd.DataFrame, label_col: str = "label") -> pd.DataFrame:
    if label_col not in elec.columns:
        return elec
    elec = elec.copy()
    elec[label_col] = elec[label_col].astype("string").str.strip()
    return elec.drop_duplicates(subset=[label_col], keep="first")


# ------------------------------------------------------------------
# Contacts
# ------------------------------------------------------------------

_REGION_COLS = ("wb.region", "ind.region", "das.region", "stein.region")


def prep_contacts(elec: pd.DataFrame, *, cml_key: str) -> pd.DataFrame:
    """Build a CML contacts-like DataFrame from BIDS electrodes.tsv
    for one coordinate space. Only the coord columns for `cml_key` are
    emitted (e.g. ``mni.x/y/z`` for `cml_key='mni'`). Region columns
    (``wb.region``/``ind.region``/``das.region``/``stein.region``) are
    preserved as-is — they are not coordinate-space-specific."""
    out = elec.copy()

    # label
    if "name" in out.columns:
        out = out.rename(columns={"name": "label"})
    if "label" not in out.columns:
        out["label"] = pd.NA
    out["label"] = out["label"].astype("string").str.strip()

    # type
    src = _find_group_col(out)
    if src is None:
        out["type"] = pd.NA
    else:
        out["type"] = out[src].astype("string").str.strip().str.lower().map(_TYPE_MAP)

    # coordinates (one space only)
    out = _assign_coords(out, cml_key)
    for axis in ("x", "y", "z"):
        c = f"{cml_key}.{axis}"
        if c not in out.columns:
            out[c] = pd.NA

    cols = ["label", f"{cml_key}.x", f"{cml_key}.y", f"{cml_key}.z", "type"]
    cols += [c for c in _REGION_COLS if c in out.columns]
    return out[cols]


# ------------------------------------------------------------------
# Pairs
# ------------------------------------------------------------------

def prep_pairs(
    elec: pd.DataFrame,
    ch_bip: pd.DataFrame,
    *,
    cml_key: str,
    label_col_channels: str = "name",
    elec_name_col: str = "name",
    region_mismatch_value: str = "mismatch",
) -> pd.DataFrame:
    """Build a CML pairs-like DataFrame from electrodes + bipolar channels
    for one coordinate space (keyed by `cml_key`). Coordinates are the
    midpoint of each pair's two contacts, emitted as {cml_key}.x/y/z.
    Region columns (wb/ind/das/stein) are preserved as-is when present
    in electrodes."""

    # Normalize electrodes
    elec2 = elec.copy()
    if elec_name_col in elec2.columns and elec_name_col != "label":
        elec2 = elec2.rename(columns={elec_name_col: "label"})
    if "label" not in elec2.columns:
        elec2["label"] = pd.NA
    elec2["label"] = elec2["label"].astype("string").str.strip()

    group_col = _find_group_col(elec2)
    elec2 = _assign_coords(elec2, cml_key)
    elec2 = _dedupe_electrodes(elec2, "label")
    elec_idx = elec2.set_index("label", drop=False)

    # NOTE: CML pair-level region labels (ind/das/stein/avg/dk/etc.) are
    # derived upstream by the neurorad pipeline via atlas lookup at each
    # pair's midpoint coordinate, NOT from the two contacts' region
    # labels. We can't reproduce that from contact-level BIDS data
    # without the atlas volumes, so we omit region columns from the
    # pairs DataFrame entirely. Regions remain comparable at the
    # contacts level. See RELEASE_NOTES.md in pennmem/neurorad_pipeline.
    present_region_cols: list[str] = []

    # Parse bipolar labels
    ch2 = ch_bip.copy().rename(columns={label_col_channels: "label"})
    ch2["label"] = ch2["label"].astype("string").str.strip()

    splits = ch2["label"].str.split("-", n=1, expand=True)
    ch2["contact1"] = splits[0].str.strip() if 0 in splits.columns else pd.NA
    ch2["contact2"] = splits[1].str.strip() if 1 in splits.columns else pd.NA

    e1 = elec_idx.reindex(ch2["contact1"].astype("string"))
    e2 = elec_idx.reindex(ch2["contact2"].astype("string"))

    def _mid(col):
        a = pd.to_numeric(e1.get(col), errors="coerce").to_numpy()
        b = pd.to_numeric(e2.get(col), errors="coerce").to_numpy()
        return (a + b) / 2.0

    out = pd.DataFrame({"label": ch2["label"].astype("string").values})
    for axis in ("x", "y", "z"):
        c = f"{cml_key}.{axis}"
        out[c] = _mid(c) if c in elec_idx.columns else pd.NA

    # type (underscore names match CML pairs schema: type_1/type_2)
    if group_col:
        out["type_1"] = e1[group_col].apply(_norm_group).to_numpy()
        out["type_2"] = e2[group_col].apply(_norm_group).to_numpy()
    else:
        out["type_1"] = pd.NA
        out["type_2"] = pd.NA

    # (region columns intentionally omitted — see note above.)
    # contact_1 / contact_2 are also omitted: CML stores them as integer
    # contact IDs from the localization JSON, but BIDS electrodes.tsv
    # exposes only the string label ("name"). We can't reproduce CML's
    # integer IDs from BIDS alone, so comparing them always fails. The
    # pair `label` already encodes both contact labels ("LAF1-LAF2").

    cols = [
        "label", f"{cml_key}.x", f"{cml_key}.y", f"{cml_key}.z",
        "type_1", "type_2",
    ] + present_region_cols
    return out[cols]
