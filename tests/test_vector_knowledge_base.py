"""Unit tests for the PostgreSQL vector knowledge-base adapter."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fatbb.infrastructure.kb.vector import PostgresVectorKnowledgeBase
from fatbb.domain.knowledge_base import KnowledgeBaseConfig
from rag.indexers import VectorIndexer
from rag.retrievers import VectorRetriever


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


class RecordingEmbeddingFactory:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str]] = []

    def create(self, provider: str, model: str, url: str) -> object:
        self.calls.append((provider, model, url))
        return object()


if __name__ == "__main__":
    unittest.main()
