"""The document loading port."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..models.document import Document


class DocumentLoader(ABC):
    """Transform a data source's raw records into :class:`Document` values.

    The interface says nothing about the source; a concrete loader owns its
    connection (path, dsn, …) and its row-to-document mapping.
    """

    @abstractmethod
    def load(self) -> list[Document]:
        """Return every document the source provides."""
