"""Sampler — discover documents, deduplicate, and register a random subset for labeling."""

from __future__ import annotations

import random
from pathlib import Path

from ..interfaces.dedup_store import DedupEntry, DedupStore, HashStatus
from ..utils.uri_resolver import URIResolver


class Sampler:
    """Discover files via a URIResolver, deduplicate via DedupStore, and
    sample a target number for labeling.

    Files are consumed from the resolver's iterator in batches so the entire
    corpus is never materialised in memory.  Batch processing stops as soon as
    enough unique clusters have been accumulated.

    Three layers of deduplication — all through the ``DedupStore`` ABC:

    1. **Existing** — hashes already ACCEPTED / REJECTED / IN_FLIGHT in the
       persistent store are skipped (``lookup_batch``).
    2. **Near-duplicate clustering** — fresh files are checked against an
       in-memory instance (``create_in_memory()``).  Only one file per
       near-duplicate cluster is kept.
    3. **Random sampling** — representatives are shuffled and drawn up to
       the target count.
    """

    def __init__(
        self,
        resolver: URIResolver,
        dedup: DedupStore,
        *,
        seed: int | None = None,
        batch_size: int = 200,
    ) -> None:
        self._resolver = resolver
        self._dedup = dedup
        self._rng = random.Random(seed)
        self._batch_size = batch_size

    # ---- public API -----------------------------------------------------------

    def sample(
        self,
        root: Path | str,
        target: int,
        *,
        holdout: int = 0,
        glob: str = "**/*.md",
    ) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
        """Walk *root*, deduplicate in batches, sample, and batch-register.

        Returns:
            ``(labeling_records, holdout_records, report)``.
        """
        total_needed = target + holdout
        file_iter = self._resolver.iter_files(root, glob=glob)

        # In-memory dedup store — same algorithm as persistent, separate state
        mem_dedup = self._dedup.create_in_memory()

        # Unique representatives: (uri, source_id, hash) — one per cluster
        representatives: list[tuple[str, str, str]] = []

        total_scanned = 0
        total_skipped_by_persistent = 0
        total_skipped_by_duplicate = 0

        while True:
            batch = self._next_batch(file_iter)
            if not batch:
                break
            total_scanned += len(batch)

            # Resolve content, compute hashes
            resolved: list[tuple[str, str, str]] = []
            for uri, sid in batch:
                text = self._resolver.resolve(uri)
                h = self._dedup.recipe_card_hash(text)
                resolved.append((uri, sid, h))

            # Layer 1: filter by persistent dedup store (already processed)
            hashes = [h for _, _, h in resolved]
            existing = self._dedup.lookup_batch(hashes)
            fresh: list[tuple[str, str, str]] = []
            for item in resolved:
                if existing.get(item[2]) is None:
                    fresh.append(item)
                else:
                    total_skipped_by_persistent += 1

            # Layer 2: filter by in-memory near-duplicate clustering
            for uri, sid, h in fresh:
                if mem_dedup.lookup(h) is not None:
                    total_skipped_by_duplicate += 1
                    continue
                # New unique cluster — register in memory and keep
                mem_dedup.register(h, HashStatus.IN_FLIGHT, source_id=sid)
                representatives.append((uri, sid, h))

            if len(representatives) >= total_needed:
                break

        if len(representatives) < total_needed:
            raise ValueError(
                f"not enough unique clusters after scanning {total_scanned} files: "
                f"need {total_needed}, have {len(representatives)}"
            )

        # Layer 3: random sample from representatives
        labeling_records, holdout_records = self._select(
            representatives, target, holdout,
        )

        # Batch-register the selected labeling records
        if labeling_records:
            self._dedup.register_batch([
                DedupEntry(
                    recipe_card_hash=str(r["recipe_card_hash"]),
                    status=HashStatus.IN_FLIGHT,
                    source_id=str(r["source_id"]),
                )
                for r in labeling_records
            ])

        report: dict[str, object] = {
            "total_scanned": total_scanned,
            "skipped_by_persistent": total_skipped_by_persistent,
            "skipped_by_duplicate": total_skipped_by_duplicate,
            "unique_clusters": len(representatives),
            "labeling_count": len(labeling_records),
            "holdout_count": len(holdout_records),
        }
        return labeling_records, holdout_records, report

    # ---- internal: batching ---------------------------------------------------

    def _next_batch(
        self, iterator: object,
    ) -> list[tuple[str, str]]:
        """Pull up to ``_batch_size`` items from the resolver iterator."""
        batch: list[tuple[str, str]] = []
        for _ in range(self._batch_size):
            try:
                batch.append(next(iterator))  # type: ignore[arg-type]
            except StopIteration:
                break
        return batch

    # ---- internal: selection --------------------------------------------------

    def _select(
        self,
        representatives: list[tuple[str, str, str]],
        target: int,
        holdout: int,
    ) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        """Shuffle representatives, split into labeling and holdout sets."""
        pool = representatives.copy()
        self._rng.shuffle(pool)

        total_needed = target + holdout
        selected = pool[:total_needed]

        labeling = selected[:target]
        holdout_list = selected[target:total_needed]

        def _make_records(
            items: list[tuple[str, str, str]], split: str,
        ) -> list[dict[str, object]]:
            return [
                {
                    "id": f"{split}:{sid}",
                    "path": uri,
                    "source_id": sid,
                    "recipe_card_hash": h,
                    "split": split,
                }
                for uri, sid, h in items
            ]

        return _make_records(labeling, "labeling"), _make_records(holdout_list, "holdout")
