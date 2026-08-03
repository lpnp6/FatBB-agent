"""Abstract interfaces for the labeling pipeline.

Every interface follows dependency inversion: the orchestrator imports ONLY
these ABCs. Concrete implementations are injected at the composition root
(run.py), enabling backend swaps with zero pipeline code changes.
"""

from .checkpoint_store import CheckpointStore, ItemStatus
from .dedup_store import DedupStore, HashStatus
from .labeling_client import LabelingClient
from .prompt_builder import PromptBuilder

__all__ = ["CheckpointStore", "DedupStore", "HashStatus", "ItemStatus", "LabelingClient", "PromptBuilder"]
