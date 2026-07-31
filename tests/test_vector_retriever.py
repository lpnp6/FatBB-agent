"""Unit tests for database-independent vector retrieval."""

from __future__ import annotations

import sys
import unittest
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag.interfaces.stores import VectorSearchStore
from rag.models import RetrievalQuery, ScoredTextChunk, SourceRef, TextChunk
from rag.retrievers import VectorRetriever


class FakeVectorStore(VectorSearchStore):
    def __init__(self, matches: list[ScoredTextChunk]) -> None:
        self.matches = matches
        self.last_search: tuple[str, int, Mapping[str, object]] | None = None

    def list_chunks(self, *, filters: Mapping[str, object]) -> list[TextChunk]:
        return []

    def upsert_chunks(self, chunks: Sequence[TextChunk]) -> None:
        pass

    def delete_chunks(self, chunk_ids: Sequence[str]) -> None:
        pass

    def replace_document_chunks(
        self,
        entries: Sequence[tuple[str, Sequence[TextChunk]]],
        *,
        on_progress: Callable[[str, int, int], None] | None = None,
    ) -> None:
        pass

    async def areplace_document_chunks(
        self,
        entries: Sequence[tuple[str, Sequence[TextChunk]]],
        *,
        on_progress: Callable[[str, int, int], None] | None = None,
    ) -> None:
        self.replace_document_chunks(entries, on_progress=on_progress)

    def delete_by_document_ids(self, document_ids: Sequence[str]) -> None:
        pass

    def search(
        self,
        query_text: str,
        *,
        top_k: int,
        filters: Mapping[str, object],
    ) -> list[ScoredTextChunk]:
        self.last_search = (query_text, top_k, filters)
        return self.matches[:top_k]

    async def asearch(
        self,
        query_text: str,
        *,
        top_k: int,
        filters: Mapping[str, object],
    ) -> list[ScoredTextChunk]:
        return self.search(query_text, top_k=top_k, filters=filters)


class VectorRetrieverTests(unittest.TestCase):
    def test_wraps_vector_matches_as_evidence(self) -> None:
        chunk = TextChunk(
            id="pasta",
            document_id="guide",
            content="Pasta uses durum wheat.",
            index=0,
            source=SourceRef(document_id="guide"),
            metadata={"category": "recipe"},
        )
        store = FakeVectorStore([ScoredTextChunk(chunk=chunk, score=0.9)])
        retriever = VectorRetriever(store)

        results = retriever.retrieve(
            RetrievalQuery("pasta", top_k=2, filters={"category": "recipe"})
        )

        self.assertEqual([result.id for result in results], ["pasta"])
        self.assertEqual(results[0].metadata["retriever"], "vector")
        self.assertEqual(store.last_search, ("pasta", 2, {"category": "recipe"}))

    def test_blank_queries_do_not_call_the_store(self) -> None:
        store = FakeVectorStore([])

        self.assertEqual(VectorRetriever(store).retrieve(RetrievalQuery("  ")), [])
        self.assertIsNone(store.last_search)


if __name__ == "__main__":
    unittest.main()
