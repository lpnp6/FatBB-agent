"""Unit tests for the PostgreSQL vector knowledge-base adapter."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fatbb.infrastructure.kb.vector import PostgresVectorKnowledgeBase
from fatbb.domain.knowledge_base import KnowledgeBaseConfig
from rag.indexers import VectorIndexer
from rag.retrievers import VectorRetriever
from rag.stores import PostgresVectorSearchStore


class PostgresVectorKnowledgeBaseTests(unittest.TestCase):
    def test_builds_vector_indexer_and_retriever(self) -> None:
        factory = RecordingEmbeddingFactory()
        adapter = PostgresVectorKnowledgeBase(factory)
        config = KnowledgeBaseConfig(
            retrieval_type="vector",
            database_type="pg",
            database_url="postgresql://example",
            source_type="file_path",
            embedding_provider="ollama",
            embedding_model="nomic-embed-text",
            embedding_url="http://localhost:11434",
        )

        self.assertEqual(adapter.type, "vector")
        self.assertIsInstance(adapter.indexer(config), VectorIndexer)
        self.assertIsInstance(adapter.retriever(config), VectorRetriever)
        self.assertEqual(
            factory.calls,
            [
                ("ollama", "nomic-embed-text", "http://localhost:11434"),
                ("ollama", "nomic-embed-text", "http://localhost:11434"),
            ],
        )

    def test_migrate_uses_the_selected_model_dimension(self) -> None:
        factory = RecordingEmbeddingFactory()
        adapter = PostgresVectorKnowledgeBase(factory)
        config = KnowledgeBaseConfig(
            retrieval_type="vector", database_type="pg", database_url="postgresql://example",
            source_type="file_path", embedding_provider="ollama",
            embedding_model="nomic-embed-text", embedding_url="http://localhost:11434",
        )

        with patch.object(PostgresVectorSearchStore, "migrate") as migrate:
            adapter.migrate(config)

        migrate.assert_called_once_with(
            scripts=(
                "0001_create_rag_text_chunks.sql",
                "0002_add_rag_text_chunk_embeddings.sql",
                "0003_add_rag_text_chunk_embedding_hnsw_index.sql",
            ),
            embedding_dimension=768,
        )


class RecordingEmbeddingFactory:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str]] = []

    def create(self, provider: str, model: str, url: str) -> object:
        self.calls.append((provider, model, url))
        return object()


if __name__ == "__main__":
    unittest.main()
