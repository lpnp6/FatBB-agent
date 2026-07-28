"""The database-independent retrieval port."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..models.evidence import Evidence
from ..models.query import RetrievalQuery


class Retriever(ABC):
    """Maps a retrieval query to descending, citeable evidence."""

    @abstractmethod
    def retrieve(self, query: RetrievalQuery) -> list[Evidence]:
        """Return no more than ``query.top_k`` results in score order."""
