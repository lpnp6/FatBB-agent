"""Ollama implementation of the embedding client port."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from urllib.error import URLError
from urllib.request import Request, urlopen

from ..interfaces.client import EmbeddingClient

logger = logging.getLogger(__name__)

# Texts per HTTP request to Ollama.  Keeps payload under ~3 MB for the
# nomic-embed-text model (max 2048 tokens ≈ ~8 KB per text).
_BATCH_SIZE = 64
_MAX_RETRIES = 3


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

    def batch_embedding(
        self, texts: Sequence[str], *,
        on_progress: Callable[[str, int, int], None] | None = None,
    ) -> list[list[float]]:
        """Generate embeddings for *texts*, splitting into sub-batches."""
        if not texts:
            return []
        total = len(texts)
        sub_batches = [
            texts[i : i + _BATCH_SIZE]
            for i in range(0, total, _BATCH_SIZE)
        ]
        t0 = time.monotonic()
        logger.info(
            "Starting sync batch embedding: %d texts in %d sub-batches",
            total, len(sub_batches),
        )
        ordered: list[object] = [None] * len(sub_batches)
        completed = 0
        failed: list[int] = []

        def _on_sub_batch(count: int) -> None:
            nonlocal completed
            completed += count
            if on_progress is not None:
                on_progress("Generating embeddings", completed, total)

        for idx, batch in enumerate(sub_batches):
            try:
                ordered[idx] = self._batch_with_retry(batch, on_progress=_on_sub_batch)
            except RuntimeError:
                logger.exception("Sub-batch %d failed after all retries", idx)
                failed.append(idx)
        elapsed = time.monotonic() - t0
        if failed:
            raise RuntimeError(
                f"{len(failed)}/{len(sub_batches)} sub-batches failed "
                f"(indices: {', '.join(str(i) for i in failed)})"
            )
        logger.info(
            "Sync batch embedding complete: %d vectors in %.1fs (%.0f ms/text)",
            total, elapsed, elapsed / total * 1000,
        )
        return [vec for batch in ordered for vec in batch]  # type: ignore[misc]

    async def a_batch_embedding(
        self, texts: Sequence[str], *,
        on_progress: Callable[[str, int, int], None] | None = None,
    ) -> list[list[float]]:
        """Generate embeddings for *texts* with sub-batches sent concurrently.

        Failed sub-batches are retried automatically; other batches
        keep running and are not cancelled.
        """
        if not texts:
            return []
        total = len(texts)
        sub_batches = [
            texts[i : i + _BATCH_SIZE]
            for i in range(0, total, _BATCH_SIZE)
        ]
        t0 = time.monotonic()
        logger.info(
            "Starting async batch embedding: %d texts in %d sub-batches",
            total, len(sub_batches),
        )
        ordered: list[object] = [None] * len(sub_batches)
        pending: list[tuple[int, Sequence[str]]] = list(enumerate(sub_batches))
        completed = 0
        completed_lock = threading.Lock()

        def _on_sub_batch(count: int) -> None:
            nonlocal completed
            with completed_lock:
                completed += count
                if on_progress is not None:
                    on_progress("Generating embeddings", completed, total)

        for attempt in range(_MAX_RETRIES):
            if not pending:
                break
            if attempt > 0:
                logger.warning(
                    "Retry attempt %d/%d: %d sub-batches remaining",
                    attempt + 1, _MAX_RETRIES, len(pending),
                )
            tasks = [
                asyncio.to_thread(self._batch_request, batch, _on_sub_batch)
                for _, batch in pending
            ]
            outcomes = await asyncio.gather(*tasks, return_exceptions=True)
            still_pending: list[tuple[int, Sequence[str]]] = []
            for (idx, _), outcome in zip(pending, outcomes):
                if isinstance(outcome, BaseException):
                    logger.warning(
                        "Sub-batch %d failed (attempt %d/%d): %s",
                        idx, attempt + 1, _MAX_RETRIES, outcome,
                    )
                    still_pending.append((idx, sub_batches[idx]))
                else:
                    ordered[idx] = outcome
            pending = still_pending

        if pending:
            failed = [str(idx) for idx, _ in pending]
            raise RuntimeError(
                f"{len(pending)} sub-batches failed after "
                f"{_MAX_RETRIES} retries (indices: {', '.join(failed)})"
            )
        elapsed = time.monotonic() - t0
        logger.info(
            "Async batch embedding complete: %d vectors in %.1fs (%.0f ms/text)",
            len(texts), elapsed, elapsed / len(texts) * 1000,
        )
        return [vec for batch in ordered for vec in batch]  # type: ignore[misc]

    def _batch_with_retry(
        self, texts: Sequence[str],
        on_progress: Callable[[int], None] | None = None,
    ) -> list[list[float]]:
        """Send one batch, retrying on transient errors."""
        last: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                return self._batch_request(texts, on_progress=on_progress)
            except RuntimeError as exc:
                last = exc
                logger.warning(
                    "Batch request failed (attempt %d/%d, %d texts): %s",
                    attempt + 1, _MAX_RETRIES, len(texts), exc,
                )
        raise RuntimeError(
            f"Sub-batch failed after {_MAX_RETRIES} retries: {last}"
        ) from last

    def _batch_request(
        self, texts: Sequence[str],
        on_progress: Callable[[int], None] | None = None,
    ) -> list[list[float]]:
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
        if on_progress is not None:
            on_progress(len(result))
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
