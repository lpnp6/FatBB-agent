"""Abstract interfaces for the labeling pipeline.

Every interface follows dependency inversion: the orchestrator imports ONLY
these ABCs. Concrete implementations are injected at the composition root
(run.py), enabling backend swaps with zero pipeline code changes.
"""

from .checkpoint_store import CheckpointStore, ItemStatus
from .dedup_store import DedupEntry, DedupStore, HashStatus
from .labeling_client import LabelingClient, TransientError
from .orchestrator import Orchestrator
from .prompt_builder import PromptBuilder
from .sampler import Sampler
from .work_queue import WorkQueue

__all__ = [
    "CheckpointStore",
    "DedupEntry",
    "DedupStore",
    "HashStatus",
    "ItemStatus",
    "LabelingClient",
    "Orchestrator",
    "PromptBuilder",
    "Sampler",
    "TransientError",
    "WorkQueue",
]
