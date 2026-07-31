
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
    _EMBEDDING_DIMENSIONS = {"nomic-embed-text": 768}
    _MIGRATION_SCRIPTS = (
        "0001_create_rag_text_chunks.sql",
        "0002_add_rag_text_chunk_embeddings.sql",
        "0003_add_rag_text_chunk_embedding_hnsw_index.sql",
    )

    def __init__(self, embedding_client_factory: EmbeddingClientFactory | None = None):
        self._embedding_client_factory = (
            embedding_client_factory or ConfiguredEmbeddingClientFactory()
        )

    def check_connection(self, config: KnowledgeBaseConfig) -> None:
        """Verify PostgreSQL is reachable before source files are read."""
        self._store(config).check_connection()

    def migrate(self, config: KnowledgeBaseConfig) -> None:
        try:
            dimension = self._EMBEDDING_DIMENSIONS[config.embedding_model or ""]
        except KeyError as error:
            raise ValueError(
                f"No embedding dimension is configured for {config.embedding_model!r}."
            ) from error
        self._store(config).migrate(
            scripts=self._MIGRATION_SCRIPTS, embedding_dimension=dimension
        )

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
    
