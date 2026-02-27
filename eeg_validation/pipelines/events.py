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
        evs_cml = load_cml_events(
            self.subject, self.experiment, self.session,
            localization=self.localization, montage=self.montage,
        )

        # Load BIDS events via BIDSReader
        try:
            evs_bids = load_bids_events(self.reader)
        except Exception as e:
            return {"skipped": True, "reason": "bids_events_not_found", "error": str(e)}

        # Experiment-specific fixes
        evs_cml, evs_bids = apply_fixes(self.experiment, evs_cml, evs_bids)

        # Prep
        prep = prep_events(
            evs_cml, evs_bids,
            evs_types=self.evs_types,
            onset_as_diff=True,
            subject=self.subject,
            experiment=self.experiment,
            session=self.session,
        )

        if len(prep["evs_cml_prepped"]) == 0 or len(prep["evs_bids_prepped"]) == 0:
            return {"skipped": True, "reason": "no_matching_events"}

        # Compare
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

        # Save
        self._save_df(result.df_summary, f"df_behavior_summary_{self.session_tag}.csv")

        return {
            "result": result,
            "paths": self._output_paths(),
        }
