"""Unit tests for database-independent BM25 retrieval."""

from __future__ import annotations

import sys
import unittest
from collections.abc import Mapping, Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag.interfaces.stores import BM25SearchStore
from rag.models import Evidence, RetrievalQuery, ScoredTextChunk, SourceRef, TextChunk
from rag.retrievers import BM25Retriever
from rag.stores import PostgresTextChunkStore


class FakeBM25Store(BM25SearchStore):
    def __init__(self, chunks: list[TextChunk], scores: dict[str, float]) -> None:
        self.chunks = chunks
        self.scores = scores
        self.last_search: tuple[str, int, Mapping[str, object]] | None = None

    def list_chunks(self, *, filters: Mapping[str, object]) -> list[TextChunk]:
        return [
            chunk
            for chunk in self.chunks
            if all(chunk.metadata.get(key) == value for key, value in filters.items())
        ]

    def upsert_chunks(self, chunks: Sequence[TextChunk]) -> None:
        by_id = {chunk.id: chunk for chunk in self.chunks}
        by_id.update({chunk.id: chunk for chunk in chunks})
        self.chunks = list(by_id.values())

    def delete_chunks(self, chunk_ids: Sequence[str]) -> None:
        ids = set(chunk_ids)
        self.chunks = [chunk for chunk in self.chunks if chunk.id not in ids]

    def replace_document_chunks(
        self, document_id: str, chunks: Sequence[TextChunk]
    ) -> None:
        self.delete_by_document_ids([document_id])
        self.upsert_chunks(chunks)

    def delete_by_document_ids(self, document_ids: Sequence[str]) -> None:
        ids = set(document_ids)
        self.chunks = [chunk for chunk in self.chunks if chunk.document_id not in ids]

    def search_bm25(
        self,
        query_text: str,
        *,
        top_k: int,
        filters: Mapping[str, object],
    ) -> list[ScoredTextChunk]:
        self.last_search = (query_text, top_k, filters)
        if query_text == "saffron":
            return []
        matches = [
            ScoredTextChunk(chunk=chunk, score=self.scores[chunk.id])
            for chunk in self.list_chunks(filters=filters)
            if chunk.id in self.scores
        ]
        return sorted(matches, key=lambda match: -match.score)[:top_k]


def chunk(chunk_id: str, content: str, **metadata: object) -> TextChunk:
    return TextChunk(
        id=chunk_id,
        document_id=f"document-{chunk_id}",
        content=content,
        index=0,
        source=SourceRef(document_id=f"document-{chunk_id}"),
        metadata=metadata,
    )


class BM25RetrieverTests(unittest.TestCase):
    """Contract tests for the database-backed BM25 retriever adapter."""

    def setUp(self) -> None:
        self.store = FakeBM25Store(
            [
                chunk("pasta", "pasta pasta lemon parmesan", category="recipe"),
                chunk("soup", "tomato soup basil", category="recipe"),
                chunk("guide", "pasta cooking guide", category="guide"),
            ],
            scores={"pasta": 3.5, "guide": 1.2},
        )
        self.retriever = BM25Retriever(self.store)

    def test_returns_descending_relevant_evidence(self) -> None:
        """The retriever preserves backend ranking and wraps results as evidence."""
        results = self.retriever.retrieve(RetrievalQuery("pasta lemon", top_k=2))

        self.assertEqual([result.id for result in results], ["pasta", "guide"])
        self.assertGreater(results[0].score, results[1].score)
        self.assertEqual(results[0].kind, "text_chunk")
        self.assertEqual(results[0].metadata["retriever"], "bm25")
        self.assertEqual(self.store.last_search, ("pasta lemon", 2, {}))

    def test_applies_metadata_filters_before_scoring(self) -> None:
        """Filters and the requested result limit are forwarded to the store."""
        results = self.retriever.retrieve(
            RetrievalQuery("pasta", top_k=5, filters={"category": "recipe"})
        )

        self.assertEqual([result.id for result in results], ["pasta"])
        self.assertEqual(
            self.store.last_search,
            ("pasta", 5, {"category": "recipe"}),
        )

    def test_returns_no_evidence_for_empty_or_unmatched_query(self) -> None:
        """Blank input short-circuits and a backend miss produces no evidence."""
        self.assertEqual(self.retriever.retrieve(RetrievalQuery("   ")), [])
        self.assertEqual(self.retriever.retrieve(RetrievalQuery("saffron")), [])

    def test_query_validates_limits(self) -> None:
        """Retrieval queries reject a non-positive result limit at construction."""
        with self.assertRaises(ValueError):
            RetrievalQuery("pasta", top_k=0)

    def test_text_evidence_requires_its_structured_chunk(self) -> None:
        """Text evidence cannot be created without its citeable chunk payload."""
        with self.assertRaises(ValueError):
            Evidence(
                id="missing-chunk",
                kind="text_chunk",
                content="pasta",
                score=1.0,
            )

    def test_postgres_row_values_are_narrowed_before_model_creation(self) -> None:
        """Untyped PostgreSQL row values become validated domain model fields."""
        parsed = PostgresTextChunkStore._to_chunk(
            (
                "chunk-1",
                "document-1",
                "pasta",
                2,
                {"document_id": "document-1", "title": "Pasta"},
                0,
                5,
                {"category": "recipe"},
            )
        )

        self.assertEqual(parsed.index, 2)
        self.assertEqual(parsed.source.title, "Pasta")
        self.assertEqual(parsed.metadata, {"category": "recipe"})


if __name__ == "__main__":
    unittest.main()
