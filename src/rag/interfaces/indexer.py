"""The document indexing port."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence

from ..models.document import Document


class Indexer(ABC):
    """Persist source documents in one or more retrieval indexes."""

    @abstractmethod
    def upsert_documents(
        self, documents: Sequence[Document], *,
        on_progress: Callable[[str, int, int], None] | None = None,
    ) -> None:
        """Create or replace every indexed representation of each document."""

    @abstractmethod
    def delete_documents(self, document_ids: Sequence[str]) -> None:
        """Remove every indexed representation for the supplied document IDs."""
