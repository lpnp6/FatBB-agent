"""Unit tests for the PostgreSQL vector knowledge-base adapter."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fatbb.infrastructure.kb.vector import PostgresVectorKnowledgeBase
from rag.indexers import VectorIndexer
from rag.retrievers import VectorRetriever


class PostgresVectorKnowledgeBaseTests(unittest.TestCase):
    def test_builds_vector_indexer_and_retriever(self) -> None:
        adapter = PostgresVectorKnowledgeBase()

        self.assertEqual(adapter.type, "vector")
        self.assertIsInstance(adapter.indexer("postgresql://example"), VectorIndexer)
        self.assertIsInstance(adapter.retriever("postgresql://example"), VectorRetriever)


if __name__ == "__main__":
    unittest.main()
