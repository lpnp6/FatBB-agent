"""Unit tests for the Ollama embedding client."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag.client import OllamaEmbeddingClient


class OllamaEmbeddingClientTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.client = OllamaEmbeddingClient("nomic-embed-text")
        self.client._request = lambda text: {"embeddings": [[0.25, 0.75]]}  # type: ignore[method-assign]

    def test_embedding_normalizes_the_provider_vector(self) -> None:
        self.assertEqual(self.client.embedding("pasta"), [0.25, 0.75])

    async def test_a_embedding_uses_the_sync_request_without_blocking(self) -> None:
        self.assertEqual(await self.client.a_embedding("pasta"), [0.25, 0.75])

    def test_rejects_an_invalid_provider_response(self) -> None:
        self.client._request = lambda text: {"embeddings": [[]]}  # type: ignore[method-assign]

        with self.assertRaisesRegex(RuntimeError, "invalid embedding vector"):
            self.client.embedding("pasta")


if __name__ == "__main__":
    unittest.main()
