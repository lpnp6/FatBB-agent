"""BM25 adapter built from the existing RAG module."""

from __future__ import annotations

from rag.chunkers import MarkdownChunker
from rag.indexers import BM25Indexer
from rag.retrievers import BM25Retriever
from rag.stores import PostgresTextChunkStore


class PostgresBm25Backend:
    type = "bm25"

    def __init__(self, dsn: str):
        self._store = PostgresTextChunkStore(dsn)

    def indexer(self) -> BM25Indexer:
        return BM25Indexer(MarkdownChunker(), self._store)

    def retriever(self) -> BM25Retriever:
        return BM25Retriever(self._store)
