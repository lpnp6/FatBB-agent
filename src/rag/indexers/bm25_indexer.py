"""Document indexing pipeline for BM25-backed lexical retrieval."""

from __future__ import annotations

from collections.abc import Sequence

from ..interfaces.chunker import Chunker
from ..interfaces.indexer import Indexer
from ..interfaces.stores import BM25SearchStore
from ..models.document import Document, TextChunk


class BM25Indexer(Indexer):
    """Chunk documents and replace their records in a BM25-capable store."""

    def __init__(self, chunker: Chunker, store: BM25SearchStore):
        self._chunker = chunker
        self._store = store

    def upsert_documents(self, documents: Sequence[Document]) -> None:
        """Rebuild each document's chunks for the backend BM25 index."""
        for document in documents:
            chunks = self._chunker.chunk(document)
            self._validate_chunks(document, chunks)
            self._store.replace_document_chunks(document.id, chunks)

    def delete_documents(self, document_ids: Sequence[str]) -> None:
        """Remove indexed chunks for deleted source documents."""
        self._store.delete_by_document_ids(document_ids)

    @staticmethod
    def _validate_chunks(document: Document, chunks: Sequence[TextChunk]) -> None:
        """Reject a chunker result that would write chunks under another document."""
        if any(chunk.document_id != document.id for chunk in chunks):
            raise ValueError("chunker returned a chunk with a mismatched document_id")
