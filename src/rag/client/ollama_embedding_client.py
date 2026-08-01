"""Ollama implementation of the embedding client port."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from urllib.error import URLError
from urllib.request import Request, urlopen

from ..interfaces.client import EmbeddingClient

logger = logging.getLogger(__name__)

# Texts per HTTP request to Ollama.  Keeps payload under ~3 MB for the
# nomic-embed-text model (max 2048 tokens ≈ ~8 KB per text).
_BATCH_SIZE = 128
_MAX_RETRIES = 3

# Concurrency controls — see docs/embedding-concurrency.md for rationale.
# A single parameter governs the semaphore, worker count, and thread pool.
_MAX_CONCURRENT_REQUESTS = 4  # matches Ollama OLLAMA_NUM_PARALLEL default


class OllamaEmbeddingClient(EmbeddingClient):
    """Generate embeddings through Ollama's local ``/api/embed`` endpoint."""

    _executor: ThreadPoolExecutor | None = None

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

    @classmethod
    def _get_executor(cls) -> ThreadPoolExecutor:
        """Return a shared thread pool for blocking HTTP calls.

        Lazily created so the first client instance pays the cost; all
        instances share the same pool so total thread count is bounded.
        """
        if cls._executor is None:
            cls._executor = ThreadPoolExecutor(
                max_workers=_MAX_CONCURRENT_REQUESTS,
            )
        return cls._executor

    # ── single embedding ──────────────────────────────────────────────

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
        """Generate embeddings for *texts*, processing sub-batches serially."""
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
        for idx, batch in enumerate(sub_batches):
            try:
                ordered[idx] = self._batch_with_retry(batch)
                completed += len(batch)
            except RuntimeError:
                logger.exception("Sub-batch %d failed after all retries", idx)
                failed.append(idx)
            if on_progress is not None:
                on_progress("Generating embeddings", completed, total)
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
        """Generate embeddings concurrently via a worker pool.

        Architecture
        ────────────
        * One **worker coroutine per sub-batch** — coroutines are cheap,
          so no need to artificially limit them.
        * An **``asyncio.Semaphore``** (sized to ``_MAX_CONCURRENT_REQUESTS``)
          is the sole concurrency gate.  Workers block on it until a
          request slot opens up.
        * A **``ThreadPoolExecutor``** (also sized to the same limit)
          runs the blocking HTTP calls so no semaphore slot ever waits
          for a thread.
        * A **result queue** feeds a single collector so ``completed`` is
          only mutated from one coroutine — no locks needed.
        * Failed sub-batches are re-queued to the task queue for up to
          ``_MAX_RETRIES`` attempts.
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

        # ── queues ─────────────────────────────────────────────────
        task_queue: asyncio.Queue[tuple[int, Sequence[str], int]] = asyncio.Queue()
        for idx, batch in enumerate(sub_batches):
            task_queue.put_nowait((idx, batch, 0))  # (idx, texts, attempt)

        # Each result is a dict so the collector can distinguish
        # success / failure without unpacking ambiguous tuples.
        result_queue: asyncio.Queue[
            tuple[int, dict[str, object]]
        ] = asyncio.Queue()

        # ── shared state (single-writer: collector) ────────────────
        ordered: list[object] = [None] * len(sub_batches)
        completed = 0
        failed: list[int] = []

        # ── concurrency controls ───────────────────────────────────
        semaphore = asyncio.Semaphore(_MAX_CONCURRENT_REQUESTS)
        executor = self._get_executor()
        loop = asyncio.get_running_loop()

        n_workers = len(sub_batches)

        # ── worker ─────────────────────────────────────────────────
        async def _worker() -> None:
            while True:
                try:
                    idx, batch, attempt = task_queue.get_nowait()
                except asyncio.QueueEmpty:
                    return

                try:
                    async with semaphore:
                        result = await loop.run_in_executor(
                            executor, self._batch_request, batch,
                        )
                    await result_queue.put((idx, {
                        "status": "ok",
                        "result": result,
                        "count": len(batch),
                    }))
                except Exception as exc:
                    if attempt + 1 < _MAX_RETRIES:
                        logger.warning(
                            "Sub-batch %d failed (attempt %d/%d): %s",
                            idx, attempt + 1, _MAX_RETRIES, exc,
                        )
                        task_queue.put_nowait((idx, batch, attempt + 1))
                    else:
                        logger.error(
                            "Sub-batch %d failed after %d retries: %s",
                            idx, _MAX_RETRIES, exc,
                        )
                        await result_queue.put((idx, {
                            "status": "failed",
                            "error": exc,
                            "count": 0,
                        }))

                task_queue.task_done()

        # ── launch workers ─────────────────────────────────────────
        workers = [asyncio.create_task(_worker()) for _ in range(n_workers)]

        # ── collector (single consumer → no lock) ──────────────────
        results_to_collect = len(sub_batches)
        while results_to_collect > 0:
            idx, entry = await result_queue.get()
            status = entry["status"]
            if status == "ok":
                ordered[idx] = entry["result"]
                completed += int(entry["count"])  # type: ignore[arg-type]
                if on_progress is not None:
                    on_progress("Generating embeddings", completed, total)
            else:
                failed.append(idx)
            results_to_collect -= 1

        # Ensure workers are done (they should be, but be safe).
        await asyncio.gather(*workers)

        if failed:
            raise RuntimeError(
                f"{len(failed)} sub-batches failed after "
                f"{_MAX_RETRIES} retries (indices: {', '.join(str(i) for i in failed)})"
            )

        elapsed = time.monotonic() - t0
        logger.info(
            "Async batch embedding complete: %d vectors in %.1fs (%.0f ms/text)",
            len(texts), elapsed, elapsed / len(texts) * 1000,
        )
        return [vec for batch in ordered for vec in batch]  # type: ignore[misc]

    # ── internal batch helpers ──────────────────────────────────────

    def _batch_with_retry(self, texts: Sequence[str]) -> list[list[float]]:
        """Send one batch, retrying on transient errors (sync path)."""
        last: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                return self._batch_request(texts)
            except RuntimeError as exc:
                last = exc
                logger.warning(
                    "Batch request failed (attempt %d/%d, %d texts): %s",
                    attempt + 1, _MAX_RETRIES, len(texts), exc,
                )
        raise RuntimeError(
            f"Sub-batch failed after {_MAX_RETRIES} retries: {last}"
        ) from last

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
