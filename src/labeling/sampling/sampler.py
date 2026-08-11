"""Sampler — discover documents, deduplicate, and stream unique items for labeling."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from pathlib import Path

from ..interfaces.checkpoint_store import CheckpointStore
from ..interfaces.dedup_store import DedupStore, HashStatus
from ..interfaces.sampler import Sampler as BaseSampler
from ..utils.uri_resolver import URIResolver


class Sampler(BaseSampler):
    """Discover files via a URIResolver, deduplicate via DedupStore, and stream
    unique items in batches.

    Files are consumed from the resolver's iterator in batches so the entire
    corpus is never materialised in memory.

    Two layers of deduplication:

    1. **Existing** — hashes already ACCEPTED / REJECTED / IN_FLIGHT in the
       persistent store are skipped (``lookup_batch``).
    2. **Near-duplicate clustering** — fresh items are checked against an
       in-memory instance (``create_in_memory()``).  Only one item per
       near-duplicate cluster is kept.
    """

    def __init__(
        self,
        resolver: URIResolver,
        dedup: DedupStore,
        checkpoint: CheckpointStore,
        *,
        batch_size: int = 200,
    ) -> None:
        self._resolver = resolver
        self._dedup = dedup
        self._checkpoint = checkpoint
        self._batch_size = batch_size

    # ---- public API -----------------------------------------------------------

    async def sample(
        self,
        root: Path | str,
        *,
        glob: str = "**/*.md",
    ) -> AsyncIterator[list[tuple[str, str, str]]]:
        """Yield batches of unique ``(source_id, hash, raw_text)`` tuples.

        Each batch is deduplicated against the persistent store (via
        ``lookup_batch``) and against items already yielded in this run
        (in-memory near-duplicate clustering).  Only the first item in each
        near-duplicate cluster is kept.

        Checkpoint state is checked before resolving each source. Completed
        (accepted) and rejected items are skipped. Interrupted IN_FLIGHT
        items are reset to PENDING and skipped for this run.
        """
        file_iter = self._resolver.iter_files(root, glob=glob)
        mem_dedup = self._dedup.create_in_memory()

        while True:
            batch = self._next_batch(file_iter)
            if not batch:
                break

            # Checkpoint filter: only resolve content for items that are
            # PENDING (new or reset).  IN_FLIGHT items are reset to PENDING
            # in the same call but excluded — they'll be picked up next run.
            pending = await self._checkpoint.select_pending(batch)
            resolved: list[tuple[str, str, str]] = []
            for sid in pending:
                text = self._resolver.resolve(sid)
                h = self._dedup.recipe_card_hash(text)
                resolved.append((sid, h, text))

            # Layer 1: filter by persistent dedup store.
            hashes = [h for _, h, _ in resolved]
            existing = self._dedup.lookup_batch(hashes)

            # Layer 2: filter by in-memory near-duplicate clustering.
            fresh: list[tuple[str, str, str]] = []
            for sid, h, text in resolved:
                if existing.get(h) is not None:
                    continue
                if mem_dedup.lookup(h) is not None:
                    continue
                mem_dedup.register(h, HashStatus.IN_FLIGHT, source_id=sid)
                fresh.append((sid, h, text))

            if fresh:
                yield fresh

    # ---- internal ------------------------------------------------------------

    def _next_batch(self, iterator: Iterator[str]) -> list[str]:
        """Pull up to ``_batch_size`` source_ids from *iterator*.

        Returns an empty list when the iterator is exhausted.
        """
        batch: list[str] = []
        for _ in range(self._batch_size):
            try:
                batch.append(next(iterator))
            except StopIteration:
                break
        return batch
