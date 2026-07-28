"""Database ports required by text retrievers and indexers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ..models.document import TextChunk


@dataclass(frozen=True)
class ScoredTextChunk:
    """A text chunk and the relevance score computed by a search backend."""

    chunk: TextChunk
    score: float


class TextChunkStore(ABC):
    """Persistence boundary for text chunks and their lifecycle operations."""

    @abstractmethod
    def list_chunks(self, *, filters: Mapping[str, object]) -> list[TextChunk]:
        """Return chunks allowed by the supplied equality filters."""

    @abstractmethod
    def upsert_chunks(self, chunks: Sequence[TextChunk]) -> None:
        """Create or replace chunks by stable chunk ID."""

    @abstractmethod
    def delete_chunks(self, chunk_ids: Sequence[str]) -> None:
        """Delete chunks by ID; unknown IDs are ignored."""


class BM25SearchStore(TextChunkStore):
    """A store capable of executing BM25 search in its native backend."""

    @abstractmethod
    def search_bm25(
        self,
        query_text: str,
        *,
        top_k: int,
        filters: Mapping[str, object],
    ) -> list[ScoredTextChunk]:
        """Return backend-scored, descending BM25 matches."""
