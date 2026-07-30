"""PostgreSQL BM25 knowledge-base adapter built from the RAG module."""

from __future__ import annotations

from rag.chunkers import MarkdownChunker
from rag.indexers import BM25Indexer
from rag.retrievers import BM25Retriever
from rag.stores import PostgresTextChunkStore


class PostgresBm25KnowledgeBase:
    type = "bm25"

    def check_connection(self, database_url: str) -> None:
        """Verify PostgreSQL is reachable before source files are read."""
        PostgresTextChunkStore(database_url).check_connection()

    def indexer(self, database_url: str) -> BM25Indexer:
        return BM25Indexer(MarkdownChunker(), PostgresTextChunkStore(database_url))

    def retriever(self, database_url: str) -> BM25Retriever:
        return BM25Retriever(PostgresTextChunkStore(database_url))
