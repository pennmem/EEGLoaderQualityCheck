"""Parity test: eeg_validation.preparers._neurorad_algo.pair_coordinate
vs. the upstream neurorad pipeline's Localization.get_pair_coordinate.

neurorad_pipeline is NOT a runtime dependency of eeg-validation. It is
loaded here dynamically from its rhino install (default:
/home2/iped/neurorad_pipeline/) for test purposes only. If the
directory isn't present (e.g. we're off rhino), the whole module is
skipped.

When the upstream get_pair_coordinate changes, this test fails. Update
_neurorad_algo.pair_coordinate AND the pinned revision recorded in
_neurorad_algo.py's top-of-file docstring in the same commit.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


NEURORAD_DIR_CANDIDATES = (
    Path("/home2/iped/neurorad_pipeline"),
    Path("/home2/iped/event_creation/neurorad"),
)


def _load_upstream_localization():
    for d in NEURORAD_DIR_CANDIDATES:
        loc_py = d / "localization.py"
        if not loc_py.exists():
            continue
        sys.path.insert(0, str(d))
        try:
            spec = importlib.util.spec_from_file_location(
                f"_upstream_localization_{d.name}", str(loc_py)
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod.Localization
        except Exception:
            sys.path.pop(0)
            continue
    return None


UpstreamLocalization = _load_upstream_localization()
pytestmark = pytest.mark.skipif(
    UpstreamLocalization is None,
    reason="upstream neurorad_pipeline not available (off rhino?)",
)


def _make_loc_with_two_contacts(space, c1_xyz, c2_xyz, ctype="raw"):
    loc = UpstreamLocalization()
    loc._contact_dict = {
        "leads": {
            "LA": {
                "type": "D",
                "n_groups": 1,
                "contacts": [
                    {
                        "name": "LA1",
                        "lead_group": 0,
                        "lead_loc": 0,
                        "grid_group": 0,
                        "grid_loc": [0, 0],
                        "atlases": {},
                        "info": {},
                        "coordinate_spaces": {space: {ctype: list(c1_xyz)}},
                    },
                    {
                        "name": "LA2",
                        "lead_group": 0,
                        "lead_loc": 1,
                        "grid_group": 0,
                        "grid_loc": [1, 0],
                        "atlases": {},
                        "info": {},
                        "coordinate_spaces": {space: {ctype: list(c2_xyz)}},
                    },
                ],
                "pairs": [],
            }
        }
    }
    return loc


SPACE_TYPE_CASES = [
    ("ct_voxel", "raw"),
    ("fs", "raw"),
    ("fs", "corrected"),
    ("t1_mri", "raw"),
    ("t2_mri", "raw"),
    ("mni", "raw"),
    ("mni", "corrected"),
]

COORD_CASES = [
    ([0.0, 0.0, 0.0], [2.0, 4.0, 6.0]),
    ([-1.5, 0.0, 10.25], [1.5, 0.0, -10.25]),
    ([100.0, 200.0, -50.0], [101.0, 201.0, -49.0]),
]


@pytest.mark.parametrize("space,ctype", SPACE_TYPE_CASES)
@pytest.mark.parametrize("c1,c2", COORD_CASES)
def test_pair_coordinate_matches_upstream(space, ctype, c1, c2):
    """pair_coordinate(c1, c2) == Localization.get_pair_coordinate
    across every (space, coordinate_type) combination."""
    from eeg_validation.preparers._neurorad_algo import pair_coordinate

    loc = _make_loc_with_two_contacts(space, c1, c2, ctype=ctype)
    upstream = loc.get_pair_coordinate(space, ["LA1", "LA2"], ctype)

    ours = pair_coordinate(c1, c2)

    np.testing.assert_allclose(
        np.asarray(upstream).ravel(), np.asarray(ours).ravel()
    )


def test_pair_coordinate_none_in_none_out():
    """Matches upstream short-circuit behavior (localization.py:286-287)."""
    from eeg_validation.preparers._neurorad_algo import pair_coordinate

    assert pair_coordinate(None, [1.0, 2.0, 3.0]) is None
    assert pair_coordinate([1.0, 2.0, 3.0], None) is None
