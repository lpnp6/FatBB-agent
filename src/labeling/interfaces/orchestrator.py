"""Orchestrator — abstract interface for the labeling execution loop."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class Orchestrator(ABC):
    """Drive the complete labeling pipeline: discover, deduplicate, label, persist.

    Concrete implementations decide the strategy:
        - Bootstrap:  cloud-model labeling + repair, writes training JSONL.
        - SFT:         fine-tuned local-model labeling, no repair, writes to DB.
    """

    @abstractmethod
    async def run(
        self,
        root: Path | str,
        target: int,
        *,
        holdout: int = 0,
        glob: str = "**/*.md",
    ) -> dict[str, Any]:
        """Run the complete pipeline and return outcome counts and metadata.

        Args:
            root: Corpus root directory or URI prefix.
            target: Minimum number of unique items to label.
            holdout: Additional items reserved for evaluation (not labeled).
            glob: File-matching pattern for discovery.

        Returns:
            Dict with at least ``"outcomes"`` (label → count mapping) and
            implementation-specific metadata.
        """
        ...
