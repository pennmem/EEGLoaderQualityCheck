"""Events (behavioral) comparison pipeline."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from .base import BasePipeline
from ..loaders.cml import load_cml_events
from ..loaders.bids import load_bids_events
from ..preparers.events import prep_events
from ..preparers.fixes import apply_fixes
from ..comparators.dataframe import DataFrameComparator


class EventsPipeline(BasePipeline):
    """Compare CML vs BIDS behavioral events for one session."""

    def __init__(
        self,
        *args,
        evs_types: Optional[Sequence[str]] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.evs_types = evs_types

    def _output_paths(self) -> List[str]:
        return [self._make_path("df_behavior_summary")]

    def _run(self) -> Dict[str, Any]:
        # Load CML events
        self._vprint(f"  Loading CML events...")
        evs_cml = load_cml_events(
            self.subject, self.experiment, self.session,
            localization=self.localization, montage=self.montage,
        )
        self._vprint(f"  CML events loaded: {len(evs_cml)} rows, types={sorted(evs_cml['type'].dropna().unique())}")

        # Load BIDS events via BIDSReader
        self._vprint(f"  Loading BIDS events...")
        try:
            evs_bids = load_bids_events(reader=self.reader, event_type=self.reader.device)
        except Exception as e:
            self._vprint(f"  BIDS events not found: {e}")
            return {"skipped": True, "reason": "bids_events_not_found", "error": str(e)}
        self._vprint(f"  BIDS events loaded: {len(evs_bids)} rows")

        # Experiment-specific fixes
        self._vprint(f"  Applying experiment-specific fixes for '{self.experiment}'...")
        evs_cml, evs_bids = apply_fixes(self.experiment, evs_cml, evs_bids)
        self._vprint(f"  After fixes: CML={len(evs_cml)} rows, BIDS={len(evs_bids)} rows")

        # Prep
        self._vprint(f"  Preparing events (evs_types={self.evs_types})...")
        prep = prep_events(
            evs_cml, evs_bids,
            evs_types=self.evs_types,
            onset_as_diff=True,
            subject=self.subject,
            experiment=self.experiment,
            session=self.session,
        )
        self._vprint(f"  Prepped: CML={len(prep['evs_cml_prepped'])} rows, BIDS={len(prep['evs_bids_prepped'])} rows")

        if len(prep["evs_cml_prepped"]) == 0 or len(prep["evs_bids_prepped"]) == 0:
            self._vprint(f"  Skipped: no matching events after prep")
            return {"skipped": True, "reason": "no_matching_events"}

        # Compare
        self._vprint(f"  Comparing CMLReader vs OpenBIDS events...")
        comparator = DataFrameComparator(
            tolerant_numeric=True,
            sort_keys=["sample", "trial_type"],
        )
        result = comparator.compare(
            prep["evs_cml_prepped"],
            prep["evs_bids_prepped"],
            label_a="CMLReader",
            label_b="OpenBIDS",
            subject=self.subject,
            experiment=self.experiment,
            session=self.session,
            return_aligned=True,
        )
        self._vprint(f"  Comparison complete (match={result.ok})")

        # Save
        self._save_df(result.df_summary, f"df_behavior_summary_{self.session_tag}.csv")

        return {
            "result": result,
            "paths": self._output_paths(),
        }
