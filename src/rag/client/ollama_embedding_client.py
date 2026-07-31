"""Ollama implementation of the embedding client port."""

from __future__ import annotations

import asyncio
import json
import math
from collections.abc import Mapping, Sequence
from urllib.error import URLError
from urllib.request import Request, urlopen

from ..interfaces.client import EmbeddingClient

# Texts per HTTP request to Ollama.  Keeps payload under ~3 MB for the
# nomic-embed-text model (max 2048 tokens ≈ ~8 KB per text).
_BATCH_SIZE = 50


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
        return self._parse_embedding(self._request(text=text))

    async def a_embedding(self, text: str) -> list[float]:
        """Generate one embedding without blocking the event loop."""
        return await asyncio.to_thread(self.embedding, text)

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

    # ── batch embedding ──────────────────────────────────────────────

    def batch_embedding(self, texts: Sequence[str]) -> list[list[float]]:
        """Generate embeddings for *texts*, splitting into sub-batches."""
        if not texts:
            return []
        results: list[list[float]] = []
        for i in range(0, len(texts), _BATCH_SIZE):
            sub = texts[i : i + _BATCH_SIZE]
            results.extend(self._batch_request(sub))
        return results

    async def a_batch_embedding(self, texts: Sequence[str]) -> list[list[float]]:
        """Generate embeddings for *texts* with sub-batches sent concurrently."""
        if not texts:
            return []
        tasks = [
            asyncio.to_thread(self._batch_request, texts[i : i + _BATCH_SIZE])
            for i in range(0, len(texts), _BATCH_SIZE)
        ]
        batches: list[list[list[float]]] = await asyncio.gather(*tasks)
        return [vec for batch in batches for vec in batch]

    def _batch_request(self, texts: Sequence[str]) -> list[list[float]]:
        """Send one batch request and return validated embedding vectors."""
        response = self._request(payload=json.dumps({"model": self._model, "input": list(texts)}))
        if not isinstance(response, Mapping):
            raise RuntimeError("Ollama returned an invalid batch embedding response")
        embeddings = response.get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != len(texts):
            raise RuntimeError(
                f"Ollama returned {len(embeddings) if isinstance(embeddings, list) else 0} "
                f"embeddings, expected {len(texts)}"
            )
        result: list[list[float]] = []
        for item in embeddings:
            if not isinstance(item, list) or not item or any(
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                for value in item
            ):
                raise RuntimeError("Ollama returned an invalid embedding vector in batch")
            result.append([float(value) for value in item])
        return result

    # ── request helpers ──────────────────────────────────────────────

    def _request(self, payload: str | None = None, text: str | None = None) -> object:
        """Send an ``/api/embed`` request with either a raw JSON *payload*
        (batch mode) or a single *text* string (single mode).
        """
        if payload is not None:
            data = payload.encode()
        elif text is not None:
            data = json.dumps({"model": self._model, "input": text}).encode()
        else:
            raise ValueError("Either payload or text must be provided")
        request = Request(
            f"{self._base_url}/api/embed",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self._timeout) as resp:
                return json.loads(resp.read())
        except URLError as error:
            raise RuntimeError(f"Ollama embedding request failed: {error.reason}") from error
