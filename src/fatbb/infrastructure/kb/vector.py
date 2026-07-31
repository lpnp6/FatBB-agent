
"""PostgreSQL pgvector knowledge-base adapter."""

from __future__ import annotations

from fatbb.domain.knowledge_base import KnowledgeBaseConfig
from fatbb.domain.ports import EmbeddingClientFactory, KnowledgeBaseAdapter
from fatbb.infrastructure.embedding import ConfiguredEmbeddingClientFactory
from rag.chunkers import MarkdownChunker
from rag.indexers import VectorIndexer
from rag.retrievers import VectorRetriever
from rag.stores import PostgresVectorSearchStore


class PostgresVectorKnowledgeBase(KnowledgeBaseAdapter):
    """Compose Ollama embeddings with PostgreSQL pgvector retrieval."""

    type = "vector"

    def __init__(self, embedding_client_factory: EmbeddingClientFactory | None = None):
        self._embedding_client_factory = (
            embedding_client_factory or ConfiguredEmbeddingClientFactory()
        )

    def check_connection(self, config: KnowledgeBaseConfig) -> None:
        """Verify PostgreSQL is reachable before source files are read."""
        self._store(config).check_connection()

    def indexer(self, config: KnowledgeBaseConfig) -> VectorIndexer:
        """Build an indexer whose store generates Ollama embeddings on write."""
        return VectorIndexer(MarkdownChunker(), self._store(config))

    def retriever(self, config: KnowledgeBaseConfig) -> VectorRetriever:
        """Build a retriever that queries PostgreSQL pgvector."""
        return VectorRetriever(self._store(config))

    def _store(self, config: KnowledgeBaseConfig) -> PostgresVectorSearchStore:
        if not config.embedding_provider:
            raise ValueError("Vector retrieval requires an embedding provider.")
        if not config.embedding_model:
            raise ValueError("Vector retrieval requires an embedding model.")
        if not config.embedding_url:
            raise ValueError("Vector retrieval requires an embedding provider URL.")
        return PostgresVectorSearchStore(
            config.database_url,
            self._embedding_client_factory.create(
                config.embedding_provider,
                config.embedding_model,
                config.embedding_url,
            ),
        )
    
