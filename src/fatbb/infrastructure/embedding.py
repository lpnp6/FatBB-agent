"""Infrastructure composition for configured embedding providers."""

from __future__ import annotations

from fatbb.domain.ports import EmbeddingClientFactory
from rag.client import OllamaEmbeddingClient
from rag.interfaces.client import EmbeddingClient


class ConfiguredEmbeddingClientFactory(EmbeddingClientFactory):
    """Create clients for embedding providers available in this deployment."""

    def create(self, provider: str, model: str, url: str) -> EmbeddingClient:
        if provider == "ollama":
            return OllamaEmbeddingClient(model, base_url=url)
        raise ValueError(f"Unsupported embedding provider: {provider}")
