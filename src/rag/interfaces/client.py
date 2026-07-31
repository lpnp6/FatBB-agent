"""Interfaces for services that turn text into embedding vectors."""

from __future__ import annotations

from abc import ABC, abstractmethod


class EmbeddingClient(ABC):
    """Generate vector embeddings from text."""

    @abstractmethod
    def embedding(self, text: str) -> list[float]:
        """Generate an embedding vector synchronously."""

    @abstractmethod
    async def a_embedding(self, text: str) -> list[float]:
        """Generate an embedding vector asynchronously."""
