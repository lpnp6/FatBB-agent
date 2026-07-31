"""Document indexing pipeline for vector-backed retrieval."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from ..interfaces.chunker import Chunker
from ..interfaces.indexer import Indexer
from ..interfaces.stores import VectorSearchStore
from ..models.document import Document, TextChunk


class VectorIndexer(Indexer):
    """Chunk documents and replace their vectors in a vector-capable store."""

    def __init__(self, chunker: Chunker, store: VectorSearchStore):
        self._chunker = chunker
        self._store = store

    def upsert_documents(
        self,
        documents: Sequence[Document],
        *,
        on_progress: Callable[[str, int, int], None] | None = None,
    ) -> None:
        """Replace each document's chunks; the store generates their embeddings."""
        total = len(documents)
        entries: list[tuple[str, Sequence[TextChunk]]] = []
        for idx, document in enumerate(documents):
            if on_progress is not None:
                on_progress("Chunking documents", idx + 1, total)
            chunks = self._chunker.chunk(document)
            if any(chunk.document_id != document.id for chunk in chunks):
                raise ValueError("chunker returned a chunk with a mismatched document_id")
            entries.append((document.id, chunks))
        self._store.replace_document_chunks(entries, on_progress=on_progress)

    def delete_documents(self, document_ids: Sequence[str]) -> None:
        """Remove every vector associated with the supplied documents."""
        self._store.delete_by_document_ids(document_ids)
