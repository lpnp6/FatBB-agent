"""Unit tests for pgvector query construction and result mapping."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag.interfaces.client import EmbeddingClient
from rag.stores.postgres.postgres_vector_search_store import PostgresVectorSearchStore


class FakeEmbeddingClient(EmbeddingClient):
    def embedding(self, text: str) -> list[float]:
        return [0.25, 0.75]

    async def a_embedding(self, text: str) -> list[float]:
        return [0.25, 0.75]

    def batch_embedding(self, texts):
        return [[0.25, 0.75] for _ in texts]

    async def a_batch_embedding(self, texts):
        return [[0.25, 0.75] for _ in texts]


class RecordingCursor:
    def __init__(self) -> None:
        self.query: object = ""
        self.parameters: list[object] = []

    def __enter__(self) -> RecordingCursor:
        return self

    def __exit__(self, *args: object) -> None:
        pass

    def execute(self, query: object, parameters: list[object]) -> None:
        self.query = query
        self.parameters = parameters

    def fetchall(self) -> list[tuple[object, ...]]:
        return [
            (
                "chunk-1", "document-1", "Pasta", 0,
                {"document_id": "document-1"}, None, None,
                {"category": "recipe"}, 0.8,
            )
        ]


class RecordingConnection:
    def __init__(self, cursor: RecordingCursor) -> None:
        self._cursor = cursor

    def __enter__(self) -> RecordingConnection:
        return self

    def __exit__(self, *args: object) -> None:
        pass

    def cursor(self) -> RecordingCursor:
        return self._cursor


class PostgresVectorSearchStoreTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.cursor = RecordingCursor()
        self.store = PostgresVectorSearchStore(
            "postgresql://example", FakeEmbeddingClient()
        )
        self.store._connect = lambda: RecordingConnection(self.cursor)  # type: ignore[method-assign]

    def test_search_filters_and_scores_in_postgres(self) -> None:
        results = self.store.search(
            "pasta", top_k=2, filters={"category": "recipe"}
        )

        self.assertEqual([result.chunk.id for result in results], ["chunk-1"])
        self.assertEqual(results[0].score, 0.8)
        self.assertIn("embedding <=> %s::vector", str(self.cursor.query))
        self.assertEqual(
            self.cursor.parameters,
            ["[0.25,0.75]", '{"category": "recipe"}', "[0.25,0.75]", 2],
        )

    def test_custom_table_name_is_used_in_query(self) -> None:
        store = PostgresVectorSearchStore(
            "postgresql://example", FakeEmbeddingClient(), table_name="tenant_chunks"
        )
        store._connect = lambda: RecordingConnection(self.cursor)  # type: ignore[method-assign]

        store.search("pasta", top_k=1, filters={})

        self.assertIn("Identifier('tenant_chunks')", str(self.cursor.query))

    def test_rejects_an_unsafe_table_name(self) -> None:
        with self.assertRaises(ValueError):
            PostgresVectorSearchStore(
                "postgresql://example",
                FakeEmbeddingClient(),
                table_name="chunks; DROP TABLE x",
            )

    async def test_asearch_uses_async_embedding_client(self) -> None:
        results = await self.store.asearch("pasta", top_k=1, filters={})

        self.assertEqual(results[0].chunk.id, "chunk-1")


if __name__ == "__main__":
    unittest.main()
