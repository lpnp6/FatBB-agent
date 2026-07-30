"""Database ports required by text retrievers and indexers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence

from ..models.document import ScoredTextChunk, TextChunk


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

    @abstractmethod
    def replace_document_chunks(
        self, entries: Sequence[tuple[str, Sequence[TextChunk]]], *,
        on_progress: Callable[[str, int, int], None] | None = None,
    ) -> None:
        """Atomically replace chunks for one or more documents in one connection."""

    @abstractmethod
    def delete_by_document_ids(self, document_ids: Sequence[str]) -> None:
        """Delete every chunk belonging to the supplied document IDs."""


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
