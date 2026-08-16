"""Database ports required by text retrievers and indexers."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence

from ..models.document import ScoredTextChunk, TextChunk
from ..models.graph import GraphEdge, GraphNode, GraphPath, ScoredGraphNode


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
        self,
        entries: Sequence[tuple[str, Sequence[TextChunk]]],
        *,
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


class VectorSearchStore(TextChunkStore):
    """A store capable of executing vector search in its native backend."""

    @abstractmethod
    def search(
        self,
        query_text: str,
        *,
        top_k: int,
        filters: Mapping[str, object],
    ) -> list[ScoredTextChunk]:
        """Return backend-scored, descending vector matches."""

    @abstractmethod
    async def asearch(
        self,
        query_text: str,
        *,
        top_k: int,
        filters: Mapping[str, object],
    ) -> list[ScoredTextChunk]:
        """Asynchronosly perform a vector similarity seatch."""

    @abstractmethod
    async def areplace_document_chunks(
        self,
        entries: Sequence[tuple[str, Sequence[TextChunk]]],
        *,
        on_progress: Callable[[str, int, int], None] | None = None,
    ) -> None:
        """Like :meth:`replace_document_chunks` but safe for async event loops."""


class GraphStore(ABC):
    """Persistence and query boundary for the knowledge graph.

    A single port serves both sides of the graph lifecycle:

    - **build** — ``GraphIndexer`` resolves existing entities and upserts the
      nodes/edges produced by mapping and fusion;
    - **retrieve** — ``GraphRetriever`` matches free text to entities and
      traverses paths.

    Like :class:`TextChunkStore`, methods take ``Sequence`` inputs and each call
    is autonomous (no explicit ``commit``), so adapters decide their own
    transaction boundaries.
    """

    # ---- build (GraphIndexer) ------------------------------------------

    @abstractmethod
    def resolve_entities(
        self, candidates: Sequence[GraphNode],
    ) -> list[GraphNode | None]:
        """Return the existing canonical node matching each candidate.

        Matching is exact and identity-only: a candidate matches an existing
        node when their ``id`` slugs are equal, or when the candidate's ``name``
        or any alias equals an existing node's ``name``/``aliases``. The result
        is position-aligned with ``candidates``; ``None`` means the candidate is
        new and should be created.

        Returned nodes carry the existing properties so the caller can fuse
        them with the candidate before upserting.
        """

    @abstractmethod
    def upsert_nodes(self, nodes: Sequence[GraphNode]) -> None:
        """Create or merge nodes. A node's ``id`` is its canonical slug."""

    @abstractmethod
    def upsert_edges(self, edges: Sequence[GraphEdge]) -> None:
        """Create or merge edges, keyed by (source_id, relation, target_id)."""

    # ---- retrieve (GraphRetriever) --------------------------------------

    @abstractmethod
    def match_entities(self, text: str, *, top_k: int) -> list[ScoredGraphNode]:
        """Return entities matching ``text`` by name/alias, scored descending."""

    @abstractmethod
    def query_nodes(self, entity_ids: Sequence[str]) -> list[GraphNode]:
        """Fetch nodes by canonical id; unknown ids are omitted."""

    @abstractmethod
    def query_paths(
        self,
        seeds: Sequence[str],
        *,
        relation_types: Sequence[str],
        max_hops: int,
        top_k: int,
    ) -> list[GraphPath]:
        """Traverse from ``seeds`` up to ``max_hops`` along matching edges.

        ``relation_types`` filters the edges traversed; an empty sequence
        traverses all relationship types.
        """

