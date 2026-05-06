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
from .comparators.onsets_heartbeats import OnsetsHeartbeatsComparator
from .comparators.heartbeat_delays import HeartbeatDelaysComparator

from .pipelines.events import EventsPipeline
from .pipelines.raw_signal import RawSignalPipeline
from .pipelines.epoched import EpochedPipeline
from .pipelines.montage import MontagePipeline
from .pipelines.digital_signal import DigitalSignalPipeline
from .pipelines.heartbeat_delays import HeartbeatDelaysPipeline

__all__ = [
    "ComparisonResult",
    "DataFrameComparator",
    "SignalComparator",
    "OnsetsHeartbeatsComparator",
    "HeartbeatDelaysComparator",
    "EventsPipeline",
    "RawSignalPipeline",
    "EpochedPipeline",
    "MontagePipeline",
    "DigitalSignalPipeline",
    "HeartbeatDelaysPipeline",
]
