"""Sampler — abstract interface for document discovery and deduplicated streaming."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from pathlib import Path


class Sampler(ABC):
    """Discover documents, deduplicate, and stream unique items for labeling.

    Concrete implementations decide the discovery mechanism (filesystem, S3,
    database) and deduplication strategy, enabling backend swaps without
    pipeline code changes.
    """

    @abstractmethod
    async def sample(
        self,
        root: Path | str,
        *,
        glob: str = "**/*.md",
    ) -> AsyncIterator[list[tuple[str, str, str]]]:
        """Yield batches of unique ``(source_id, hash, raw_text)`` tuples.

        Each batch is deduplicated against the persistent store and against
        items already yielded in this run.  Only the first item in each
        near-duplicate cluster is kept.

        Args:
            root: Corpus root directory or URI prefix.
            glob: File-matching pattern for discovery.

        Returns:
            Async iterator of batches. Each batch is a list of
            ``(source_id, hash, raw_text)`` tuples.
        """
        ...
