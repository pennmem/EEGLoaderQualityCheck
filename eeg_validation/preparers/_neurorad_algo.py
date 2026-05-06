"""Pair-location algorithms copied verbatim from the CML neurorad pipeline.

Source of truth:
    github.com/pennmem/neurorad_pipeline
    File: localization.py :: Localization.get_pair_coordinate
    Pinned revision (on rhino2): cf563def3087126b1aedb1dea274172200f3fa75
    Path checked: /home2/iped/neurorad_pipeline/localization.py

We copy rather than import to keep neurorad_pipeline off the runtime
dependency list. A test in tests/test_pair_midpoint_parity.py imports
neurorad_pipeline as a dev-only dependency and pins this copy against
the upstream implementation.

When the upstream algorithm changes, update this file AND the pinned
revision in the parity test's docstring in the same commit.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


# Mirrors Localization.VALID_COORDINATE_SPACES (localization.py:13-19).
VALID_COORDINATE_SPACES: Tuple[str, ...] = (
    "ct_voxel",
    "fs",
    "t1_mri",
    "t2_mri",
    "mni",
)

# Mirrors Localization.VALID_COORDINATE_TYPES (localization.py:21-24).
VALID_COORDINATE_TYPES: Tuple[str, ...] = (
    "raw",
    "corrected",
)

# Mirrors Localization.VALID_ATLASES (localization.py:35-39).
VALID_ATLASES: Tuple[str, ...] = (
    "dk",
    "whole_brain",
    "mtl",
)

# BIDS space label -> (neurorad coordinate_space, coordinate_type).
# Synthesized from event_creation/submission/neurorad_tasks.py
# FIELD_NAMES_TABLE (lines 190-198) and the bids-convert CML_TO_BIDS_SPACE
# map. Spaces not present here (Talairach, t1MRI, fsnativeDural) are
# valid midpoint targets but have no neurorad atlas lookup.
BIDS_SPACE_TO_NEURORAD: Dict[str, Tuple[str, str]] = {
    "fsnative":            ("fs",        "raw"),
    "fsnativeBrainshift":  ("fs",        "corrected"),
    "fsaverage":           ("fsaverage", "raw"),
    "fsaverageBrainshift": ("fsaverage", "corrected"),
    "MNI152NLin6ASym":     ("mni",       "raw"),
    "Pixels":              ("ct_voxel",  "raw"),
}


def pair_coordinate(coord1, coord2):
    """Midpoint of two contact coordinates.

    Verbatim behavior of
    neurorad_pipeline.localization.Localization.get_pair_coordinate
    (localization.py:274-289), stripped of its Localization-object
    plumbing so callers can pass raw arrays.
    """
    if coord1 is None or coord2 is None:
        return None
    return (np.array(coord1) + np.array(coord2)) / 2


def pair_coordinate_axis(a, b):
    """Vectorized midpoint for one coordinate axis across N pairs.

    NaN on either side propagates, matching the None-returning behavior
    of scalar pair_coordinate. Intended for pandas/numpy column math.
    """
    return (np.asarray(a, dtype=float) + np.asarray(b, dtype=float)) / 2.0
