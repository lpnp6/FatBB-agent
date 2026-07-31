
"""PostgreSQL pgvector knowledge-base adapter."""

from __future__ import annotations

from fatbb.domain.ports import KnowledgeBaseAdapter
from rag.chunkers import MarkdownChunker
from rag.client import OllamaEmbeddingClient
from rag.indexers import VectorIndexer
from rag.retrievers import VectorRetriever
from rag.stores import PostgresVectorSearchStore


class PostgresVectorKnowledgeBase:
    """Compose Ollama embeddings with PostgreSQL pgvector retrieval."""

    type = "vector"

    def __init__(
        self,
        *,
        embedding_model: str = "nomic-embed-text",
        ollama_url: str = "http://localhost:11434",
    ) -> None:
        self._embedding_model = embedding_model
        self._ollama_url = ollama_url

    def check_connection(self, database_url: str) -> None:
        """Verify PostgreSQL is reachable before source files are read."""
        self._store(database_url).check_connection()

    def indexer(self, database_url: str) -> VectorIndexer:
        """Build an indexer whose store generates Ollama embeddings on write."""
        return VectorIndexer(MarkdownChunker(), self._store(database_url))

    def retriever(self, database_url: str) -> VectorRetriever:
        """Build a retriever that queries PostgreSQL pgvector."""
        return VectorRetriever(self._store(database_url))

    def _store(self, database_url: str) -> PostgresVectorSearchStore:
        return PostgresVectorSearchStore(
            database_url,
            OllamaEmbeddingClient(
                self._embedding_model,
                base_url=self._ollama_url,
            ),
        )
    
