"""Prepare BIDS electrodes/channels into CML-compatible contacts and pairs DataFrames."""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


_TYPE_MAP = {"grid": "G", "depth": "D", "strip": "S", "gird": "G"}

_GROUP_COL_CANDIDATES = ("type", "group", "Group", "electrode_type", "contact_type")


def _find_group_col(df: pd.DataFrame) -> Optional[str]:
    for cand in _GROUP_COL_CANDIDATES:
        if cand in df.columns:
            return cand
    return None


def _norm_group(g, type_map=_TYPE_MAP):
    if pd.isna(g):
        return pd.NA
    return type_map.get(str(g).strip().lower().replace("gird", "grid"), pd.NA)


def _assign_coords(df: pd.DataFrame, elec_space: str) -> pd.DataFrame:
    """Rename x/y/z to mni.* or tal.* based on space; fill the other with NA."""
    if elec_space == "MNI152NLin6ASym":
        df = df.rename(columns={"x": "mni.x", "y": "mni.y", "z": "mni.z"})
        for c in ("tal.x", "tal.y", "tal.z"):
            if c not in df.columns:
                df[c] = pd.NA
    elif elec_space == "Talairach":
        df = df.rename(columns={"x": "tal.x", "y": "tal.y", "z": "tal.z"})
        for c in ("mni.x", "mni.y", "mni.z"):
            if c not in df.columns:
                df[c] = pd.NA
    else:
        for c in ("mni.x", "mni.y", "mni.z", "tal.x", "tal.y", "tal.z"):
            if c not in df.columns:
                df[c] = pd.NA
    return df


def _dedupe_electrodes(elec: pd.DataFrame, label_col: str = "label") -> pd.DataFrame:
    if label_col not in elec.columns:
        return elec
    elec = elec.copy()
    elec[label_col] = elec[label_col].astype("string").str.strip()
    return elec.drop_duplicates(subset=[label_col], keep="first")


# ------------------------------------------------------------------
# Contacts
# ------------------------------------------------------------------

def prep_contacts(elec: pd.DataFrame, *, elec_space: str) -> pd.DataFrame:
    """Build a CML contacts-like DataFrame from BIDS electrodes.tsv."""
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

    # coordinates
    out = _assign_coords(out, elec_space)

    # region
    if "ind.region" not in out.columns:
        out["ind.region"] = pd.NA

    cols = ["label", "mni.x", "mni.y", "mni.z", "type", "ind.region", "tal.x", "tal.y", "tal.z"]
    for c in cols:
        if c not in out.columns:
            out[c] = pd.NA
    return out[cols]


# ------------------------------------------------------------------
# Pairs
# ------------------------------------------------------------------

def prep_pairs(
    elec: pd.DataFrame,
    ch_bip: pd.DataFrame,
    *,
    elec_space: str,
    label_col_channels: str = "name",
    elec_name_col: str = "name",
    region_col: str = "ind.region",
    region_mismatch_value: str = "mismatch",
) -> pd.DataFrame:
    """Build a CML pairs-like DataFrame from electrodes + bipolar channels."""

    # Normalize electrodes
    elec2 = elec.copy()
    if elec_name_col in elec2.columns and elec_name_col != "label":
        elec2 = elec2.rename(columns={elec_name_col: "label"})
    if "label" not in elec2.columns:
        elec2["label"] = pd.NA
    elec2["label"] = elec2["label"].astype("string").str.strip()

    group_col = _find_group_col(elec2)
    elec2 = _assign_coords(elec2, elec_space)
    if region_col not in elec2.columns:
        elec2[region_col] = pd.NA

    elec2 = _dedupe_electrodes(elec2, "label")
    elec_idx = elec2.set_index("label", drop=False)

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
    for prefix in ("mni", "tal"):
        for axis in ("x", "y", "z"):
            c = f"{prefix}.{axis}"
            out[c] = _mid(c) if c in elec_idx.columns else pd.NA

    # type
    if group_col:
        out["type1"] = e1[group_col].apply(_norm_group).to_numpy()
        out["type2"] = e2[group_col].apply(_norm_group).to_numpy()
    else:
        out["type1"] = pd.NA
        out["type2"] = pd.NA

    # region
    r1 = e1[region_col].astype("string").to_numpy() if region_col in e1.columns else np.full(len(out), pd.NA, dtype=object)
    r2 = e2[region_col].astype("string").to_numpy() if region_col in e2.columns else np.full(len(out), pd.NA, dtype=object)
    same = (r1 == r2) & (~pd.isna(r1)) & (~pd.isna(r2))
    region = np.full(len(out), pd.NA, dtype=object)
    region[same] = r1[same]
    region[(~same) & (~pd.isna(r1)) & (~pd.isna(r2))] = region_mismatch_value
    out["ind.region"] = region

    out["contact1"] = ch2["contact1"].astype("string").values
    out["contact2"] = ch2["contact2"].astype("string").values

    cols = [
        "label", "mni.x", "mni.y", "mni.z", "tal.x", "tal.y", "tal.z",
        "type1", "type2", "ind.region", "contact1", "contact2",
    ]
    for c in cols:
        if c not in out.columns:
            out[c] = pd.NA
    return out[cols]
