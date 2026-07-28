"""The document-to-chunk transformation port."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..models.document import Document, TextChunk


class Chunker(ABC):
    """Split one source document into ordered, citeable text chunks."""

    @abstractmethod
    def chunk(self, document: Document) -> list[TextChunk]:
        """Return chunks in source order with stable IDs and source references."""
