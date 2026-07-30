"""Document indexing pipeline for BM25-backed lexical retrieval."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import logging

from tqdm import tqdm

from ..interfaces.chunker import Chunker
from ..interfaces.indexer import Indexer
from ..interfaces.stores import BM25SearchStore
from ..models.document import Document, TextChunk


logger = logging.getLogger(__name__)


class BM25Indexer(Indexer):
    """Chunk documents and replace their records in a BM25-capable store."""

    def __init__(self, chunker: Chunker, store: BM25SearchStore):
        self._chunker = chunker
        self._store = store

    def upsert_documents(
        self, documents: Sequence[Document], *,
        on_progress: Callable[[str, int, int], None] | None = None,
    ) -> None:
        """Rebuild each document's chunks for the backend BM25 index."""
        total = len(documents)
        logger.info("Upserting documents into BM25 index", extra={"document_count": total})
        # Chunk every document eagerly so that the store receives a single
        # batch write in one connection, avoiding per-document TCP overhead.
        entries: list[tuple[str, Sequence[TextChunk]]] = []
        for idx, document in enumerate(
            tqdm(documents, desc="Chunking documents", unit="document", disable=on_progress is not None)
        ):
            if on_progress is not None:
                on_progress("Chunking documents", idx + 1, total)
            chunks = self._chunker.chunk(document)
            self._validate_chunks(document, chunks)
            entries.append((document.id, chunks))
        self._store.replace_document_chunks(entries, on_progress=on_progress)
        logger.info("Completed BM25 document upsert", extra={"document_count": total})

    def delete_documents(self, document_ids: Sequence[str]) -> None:
        """Remove indexed chunks for deleted source documents."""
        self._store.delete_by_document_ids(document_ids)

    @staticmethod
    def _validate_chunks(document: Document, chunks: Sequence[TextChunk]) -> None:
        """Reject a chunker result that would write chunks under another document."""
        if any(chunk.document_id != document.id for chunk in chunks):
            raise ValueError("chunker returned a chunk with a mismatched document_id")
