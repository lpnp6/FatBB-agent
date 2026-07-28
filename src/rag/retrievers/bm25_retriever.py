"""Adapter from a backend BM25 search to uniform RAG evidence."""

from __future__ import annotations

from ..interfaces.retriever import Retriever
from ..interfaces.stores import BM25SearchStore
from ..models.evidence import Evidence
from ..models.query import RetrievalQuery


class BM25Retriever(Retriever):
    """Delegate lexical ranking to a native BM25-capable storage backend."""

    def __init__(self, store: BM25SearchStore):
        self._store = store

    def retrieve(self, query: RetrievalQuery) -> list[Evidence]:
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
                    "retriever": "bm25",
                    "raw_score": match.score,
                },
                chunk=match.chunk,
            )
            for match in self._store.search_bm25(
                query.text,
                top_k=query.top_k,
                filters=query.filters,
            )
        ]
