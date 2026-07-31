"""Unit tests for database-independent vector indexing."""

from __future__ import annotations

import sys
import unittest
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag.chunkers import MarkdownChunker
from rag.indexers import VectorIndexer
from rag.interfaces.stores import VectorSearchStore
from rag.models import Document, ScoredTextChunk, SourceRef, TextChunk


class RecordingVectorStore(VectorSearchStore):
    def __init__(self) -> None:
        self.replacements: list[tuple[str, list[TextChunk]]] = []
        self.deleted_document_ids: list[str] = []

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
        self.replacements.extend((document_id, list(chunks)) for document_id, chunks in entries)

    async def areplace_document_chunks(
        self,
        entries: Sequence[tuple[str, Sequence[TextChunk]]],
        *,
        on_progress: Callable[[str, int, int], None] | None = None,
    ) -> None:
        self.replace_document_chunks(entries, on_progress=on_progress)

    def delete_by_document_ids(self, document_ids: Sequence[str]) -> None:
        self.deleted_document_ids.extend(document_ids)

    def search(
        self,
        query_text: str,
        *,
        top_k: int,
        filters: Mapping[str, object],
    ) -> list[ScoredTextChunk]:
        return []

    async def asearch(
        self,
        query_text: str,
        *,
        top_k: int,
        filters: Mapping[str, object],
    ) -> list[ScoredTextChunk]:
        return []


class VectorIndexerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = RecordingVectorStore()
        self.indexer = VectorIndexer(
            MarkdownChunker(max_chars=200, min_chunk_chars=20), self.store
        )

    def test_upsert_replaces_document_chunks(self) -> None:
        document = Document(
            id="guide",
            content="# Guide\n\nPostgreSQL supports vector retrieval.",
            source=SourceRef(uri="https://example.com/guide"),
            metadata={"tenant_id": "acme"},
        )

        self.indexer.upsert_documents([document])

        document_id, chunks = self.store.replacements[0]
        self.assertEqual(document_id, "guide")
        self.assertEqual(chunks[0].document_id, "guide")
        self.assertEqual(chunks[0].metadata["tenant_id"], "acme")

    def test_delete_forwards_document_ids(self) -> None:
        self.indexer.delete_documents(["guide"])

        self.assertEqual(self.store.deleted_document_ids, ["guide"])


if __name__ == "__main__":
    unittest.main()
