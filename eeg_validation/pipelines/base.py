"""Base pipeline with shared save/skip logic and a BIDSReader instance."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Union

import pandas as pd

from ..loaders.bids import get_reader


class BasePipeline(ABC):
    """Common skeleton for all validation pipelines.

    Each pipeline owns a ``BIDSReader`` (``self.reader``) created from
    the session parameters.  Subclasses implement :meth:`_run` and
    declare :meth:`_output_paths`.
    """

    def __init__(
        self,
        subject: str,
        experiment: str,
        session: Union[str, int],
        bids_root: str,
        out_path: str,
        *,
        skip_if_exists: bool = True,
        # CML-only (iEEG)
        localization: Optional[int] = None,
        montage: Optional[int] = None,
    ):
        self.subject = subject
        self.experiment = experiment
        self.session = session
        self.bids_root = bids_root
        self.out_path = out_path
        self.skip_if_exists = skip_if_exists
        self.localization = localization
        self.montage = montage

        # Single BIDSReader for the whole pipeline
        self.reader = get_reader(subject, experiment, session, bids_root)

    @property
    def is_intracranial(self) -> bool:
        return self.reader.is_intracranial()

    @property
    def session_tag(self) -> str:
        return f"{self.subject}_{self.experiment}_{self.session}"

    # ------------------------------------------------------------------
    # Template method
    # ------------------------------------------------------------------
    def run(self) -> Dict[str, Any]:
        os.makedirs(self.out_path, exist_ok=True)

        paths = self._output_paths()
        if self.skip_if_exists and all(os.path.exists(p) for p in paths):
            return {"skipped": True, "reason": "outputs_exist", "paths": paths}

        return self._run()

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------
    @abstractmethod
    def _output_paths(self) -> List[str]:
        """Return list of expected output file paths."""
        ...

    @abstractmethod
    def _run(self) -> Dict[str, Any]:
        """Execute the pipeline.  Return results dict."""
        ...

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _save_df(self, df: pd.DataFrame, filename: str) -> str:
        path = os.path.join(self.out_path, filename)
        df.to_csv(path, index=False)
        return path

    def _make_path(self, prefix: str, suffix: str = ".csv") -> str:
        return os.path.join(self.out_path, f"{prefix}_{self.session_tag}{suffix}")
