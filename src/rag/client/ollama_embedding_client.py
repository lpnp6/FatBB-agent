"""Ollama implementation of the embedding client port."""

from __future__ import annotations

import asyncio
import json
import math
from collections.abc import Mapping
from urllib.error import URLError
from urllib.request import Request, urlopen

from ..interfaces.client import EmbeddingClient


class OllamaEmbeddingClient(EmbeddingClient):
    """Generate embeddings through Ollama's local ``/api/embed`` endpoint."""

    def __init__(
        self,
        model: str,
        *,
        base_url: str = "http://localhost:11434",
        timeout: float = 30.0,
    ) -> None:
        if not model:
            raise ValueError("model cannot be empty")
        if not base_url:
            raise ValueError("base_url cannot be empty")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    def embedding(self, text: str) -> list[float]:
        """Generate one embedding synchronously."""
        if not text:
            raise ValueError("text cannot be empty")
        return self._parse_embedding(self._request(text))

    async def a_embedding(self, text: str) -> list[float]:
        """Generate one embedding without blocking the event loop."""
        return await asyncio.to_thread(self.embedding, text)

    def _request(self, text: str) -> object:
        request = Request(
            f"{self._base_url}/api/embed",
            data=json.dumps({"model": self._model, "input": text}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self._timeout) as response:
                return json.loads(response.read())
        except URLError as error:
            raise RuntimeError(f"Ollama embedding request failed: {error.reason}") from error

    @staticmethod
    def _parse_embedding(response: object) -> list[float]:
        if not isinstance(response, Mapping):
            raise RuntimeError("Ollama returned an invalid embedding response")
        embeddings = response.get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != 1:
            raise RuntimeError("Ollama returned no embedding for the input text")
        values = embeddings[0]
        if not isinstance(values, list) or not values or any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            for value in values
        ):
            raise RuntimeError("Ollama returned an invalid embedding vector")
        return [float(value) for value in values]
