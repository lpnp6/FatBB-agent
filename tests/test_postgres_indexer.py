"""Unit tests for the database-independent BM25 document indexing pipeline."""

from __future__ import annotations

import sys
import unittest
from collections.abc import Mapping, Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag.chunkers import MarkdownChunker
from rag.indexers import BM25Indexer
from rag.interfaces import BM25SearchStore
from rag.models import Document, ScoredTextChunk, SourceRef, TextChunk


class RecordingBM25Store(BM25SearchStore):
    """A store double that records document-level mutation requests."""

    def __init__(self) -> None:
        self.replacements: list[tuple[str, list[TextChunk]]] = []
        self.deleted_document_ids: list[str] = []

    def list_chunks(self, *, filters: Mapping[str, object]) -> list[TextChunk]:
        """Return no data because this double only verifies indexing writes."""
        return []

    def upsert_chunks(self, chunks: Sequence[TextChunk]) -> None:
        """Accept direct writes required by the storage interface."""

    def delete_chunks(self, chunk_ids: Sequence[str]) -> None:
        """Accept chunk deletion required by the storage interface."""

    def replace_document_chunks(
        self, entries: Sequence[tuple[str, Sequence[TextChunk]]], *,
        on_progress: object = None,
    ) -> None:
        """Capture the exact replacement payloads for one or more documents."""
        self.replacements.extend((document_id, list(chunks)) for document_id, chunks in entries)

    def delete_by_document_ids(self, document_ids: Sequence[str]) -> None:
        """Capture document IDs requested for deletion."""
        self.deleted_document_ids.extend(document_ids)

    def search_bm25(
        self,
        query_text: str,
        *,
        top_k: int,
        filters: Mapping[str, object],
    ) -> list[ScoredTextChunk]:
        """Return no data because this double only verifies indexing writes."""
        return []


class BM25IndexerTests(unittest.TestCase):
    """Verify document chunking is delegated to the store as full replacement."""

    def setUp(self) -> None:
        self.store = RecordingBM25Store()
        self.indexer = BM25Indexer(
            MarkdownChunker(max_chars=200, min_chunk_chars=20), self.store
        )
        self.document = Document(
            id="guide",
            content="# Guide\n\nPostgreSQL provides BM25 retrieval through pg_search.",
            source=SourceRef(uri="https://example.com/guide"),
            metadata={"tenant_id": "acme"},
        )

    def test_upsert_chunks_and_replaces_one_document_atomically(self) -> None:
        """Indexer sends the complete, source-preserving chunk set to the store."""
        self.indexer.upsert_documents([self.document])

        document_id, chunks = self.store.replacements[0]
        self.assertEqual(document_id, "guide")
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].document_id, "guide")
        self.assertEqual(chunks[0].source.document_id, "guide")
        self.assertEqual(chunks[0].metadata["tenant_id"], "acme")

    def test_delete_documents_forwards_document_ids_to_store(self) -> None:
        """Deleting source documents clears all of their indexed chunks."""
        self.indexer.delete_documents(["guide", "obsolete"])

        self.assertEqual(self.store.deleted_document_ids, ["guide", "obsolete"])


if __name__ == "__main__":
    unittest.main()
