"""Interfaces for services that turn text into embedding vectors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence


class EmbeddingClient(ABC):
    """Generate vector embeddings from text."""

    @abstractmethod
    def embedding(self, text: str) -> list[float]:
        """Generate an embedding vector synchronously."""

    @abstractmethod
    async def a_embedding(self, text: str) -> list[float]:
        """Generate an embedding vector asynchronously."""

    @abstractmethod
    def batch_embedding(
        self, texts: Sequence[str], *,
        on_progress: Callable[[str, int, int], None] | None = None,
    ) -> list[list[float]]:
        """Generate embeddings for multiple texts in one or more requests."""

    @abstractmethod
    async def a_batch_embedding(
        self, texts: Sequence[str], *,
        on_progress: Callable[[str, int, int], None] | None = None,
    ) -> list[list[float]]:
        """Generate embeddings for multiple texts asynchronously."""
