"""Adapter from backend vector search to uniform RAG evidence."""

from __future__ import annotations

from ..interfaces.retriever import Retriever
from ..interfaces.stores import VectorSearchStore
from ..models.evidence import Evidence
from ..models.query import RetrievalQuery


class VectorRetriever(Retriever):
    """Delegate vector ranking to a native vector-capable storage backend."""

    def __init__(self, store: VectorSearchStore):
        self._store = store

    def retrieve(self, query: RetrievalQuery) -> list[Evidence]:
        """Return vector-ranked chunks as citeable text evidence."""
        if not query.text.strip():
            return []

        return [
            Evidence(
                id=match.chunk.id,
                kind="text_chunk",
                content=match.chunk.content,
                score=match.score,
                source=match.chunk.source,
                metadata={
                    **match.chunk.metadata,
                    "retriever": "vector",
                    "raw_score": match.score,
                },
                chunk=match.chunk,
            )
            for match in self._store.search(
                query.text,
                top_k=query.top_k,
                filters=query.filters,
            )
        ]
