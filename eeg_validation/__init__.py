"""
eeg_validation — Compare CML and BIDS data for behavioral events,
raw/epoched EEG signals, and montage metadata.

Subpackages
-----------
comparators : Pure comparison logic (no I/O)
loaders     : Data loading helpers (CML, BIDS, EDF)
preparers   : Schema normalization and alignment
pipelines   : End-to-end orchestration (load → prep → compare → save)
"""

from .comparators.base import ComparisonResult
from .comparators.dataframe import DataFrameComparator
from .comparators.signal import SignalComparator

from .pipelines.events import EventsPipeline
from .pipelines.raw_signal import RawSignalPipeline
from .pipelines.epoched import EpochedPipeline
from .pipelines.montage import MontagePipeline
from .pipelines.digital_signal import DigitalSignalPipeline

__all__ = [
    "ComparisonResult",
    "DataFrameComparator",
    "SignalComparator",
    "EventsPipeline",
    "RawSignalPipeline",
    "EpochedPipeline",
    "MontagePipeline",
    "DigitalSignalPipeline"
]
