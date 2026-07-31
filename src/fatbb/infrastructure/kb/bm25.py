"""PostgreSQL BM25 knowledge-base adapter built from the RAG module."""

from __future__ import annotations

from fatbb.domain.knowledge_base import KnowledgeBaseConfig
from rag.chunkers import MarkdownChunker
from rag.indexers import BM25Indexer
from rag.retrievers import BM25Retriever
from rag.stores import PostgresBM25SearchStore

class PostgresBm25KnowledgeBase:
    type = "bm25"

    def check_connection(self, config: KnowledgeBaseConfig) -> None:
        """Verify PostgreSQL is reachable before source files are read."""
        PostgresBM25SearchStore(config.database_url).check_connection()

    def indexer(self, config: KnowledgeBaseConfig) -> BM25Indexer:
        return BM25Indexer(MarkdownChunker(), PostgresBM25SearchStore(config.database_url))

    def retriever(self, config: KnowledgeBaseConfig) -> BM25Retriever:
        return BM25Retriever(PostgresBM25SearchStore(config.database_url))
