"""Base classes for all comparators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import pandas as pd


@dataclass
class ComparisonResult:
    """Uniform container returned by every comparator.

    Every pipeline can inspect the same fields regardless of whether the
    comparison was on DataFrames, raw signals, or time coordinates.
    """

    # High-level verdict
    ok: bool = True

    # Summary table (one row per comparison run)
    df_summary: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Per-column or per-channel detail
    df_detail: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Mismatch examples
    df_mismatches: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Optional aligned data (for debugging / downstream use)
    aligned_a: Optional[pd.DataFrame] = None
    aligned_b: Optional[pd.DataFrame] = None

    # Session metadata (always propagated)
    subject: Optional[str] = None
    experiment: Optional[str] = None
    session: Optional[Union[str, int]] = None

    # Free-form extras
    extras: Dict[str, Any] = field(default_factory=dict)


class Comparator(ABC):
    """Interface that all concrete comparators implement."""

    @abstractmethod
    def compare(self, a, b, **kwargs) -> ComparisonResult:
        ...
