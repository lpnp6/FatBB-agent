"""Bootstrap orchestrator — discover, deduplicate, label, validate, repair, persist."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from .checkpoint import CheckpointManager
from ..interfaces.dedup_store import DedupEntry, DedupStore, HashStatus
from ..interfaces.labeling_client import LabelingClient
from ..interfaces.orchestrator import Orchestrator
from ..sampling.sampler import Sampler
from ..utils.validator import OutputValidationError, OutputValidator


logger = logging.getLogger(__name__)


class BootstrapOrchestrator(Orchestrator):
    """Complete bootstrap pipeline: discover → dedup → label → persist.

    Owns the :class:`Sampler` and drives the full lifecycle:

    1. Stream batches from :meth:`Sampler.iter_batches` (persistent dedup +
       in-memory near-duplicate clustering already applied).
    2. Filter each batch against the checkpoint — items already completed or
       failed in a prior run are skipped before any model call.
    3. Register fresh items as IN_FLIGHT in both the dedup store and
       checkpoint, then label / validate / repair concurrently.
    4. Flush dedup-store updates at each batch boundary.

    Crash recovery: on restart the sampler's persistent dedup filter skips
    already-ACCEPTED hashes, and the checkpoint filter skips completed/failed
    items.  No separate training-JSONL recovery is needed.
    """

    def __init__(
        self,
        *,
        client: LabelingClient,
        dedup_store: DedupStore,
        sampler: Sampler,
        checkpoint: CheckpointManager,
        validator: OutputValidator | None = None,
        retries: int = 2,
        batch_size: int = 50,
    ) -> None:
        self._client = client
        self._dedup_store = dedup_store
        self._sampler = sampler
        self._checkpoint = checkpoint
        self._validator = validator or OutputValidator()
        self._retries = retries
        self._batch_size = batch_size

    # ---- Orchestrator interface -------------------------------------------

    async def run(
        self,
        root: Path | str,
        target: int,
        *,
        holdout: int = 0,
        glob: str = "**/*.md",
    ) -> dict[str, Any]:
        """Run the complete pipeline — sample, checkpoint-filter, label, persist.

        Returns a dict with ``"outcomes"`` (label → count) and metadata.
        """
        await self._checkpoint.load()

        total_needed = target + holdout
        scanned = 0
        skipped_by_checkpoint = 0
        outcomes: list[str] = []

        for batch in self._sampler.sample(root, glob=glob):
            scanned += len(batch)

            # ── checkpoint filter ──────────────────────────────────────
            fresh: list[dict[str, Any]] = []
            for sid, h, text in batch:
                item_id = f"labeling:{sid}"
                try:
                    item = self._checkpoint.item(item_id)
                except KeyError:
                    item = {"status": "pending"}
                if item["status"] in ("completed", "failed"):
                    skipped_by_checkpoint += 1
                    continue
                fresh.append({
                    "id": item_id,
                    "source_id": sid,
                    "recipe_card_hash": h,
                    "raw_text": text,
                })

            if not fresh:
                continue

            # ── register IN_FLIGHT ─────────────────────────────────────
            self._dedup_store.register_batch([
                DedupEntry(
                    recipe_card_hash=str(r["recipe_card_hash"]),
                    status=HashStatus.IN_FLIGHT,
                    source_id=str(r["source_id"]),
                    raw_text=str(r["raw_text"]),
                )
                for r in fresh
            ])
            await self._checkpoint.ensure_items(
                [str(r["id"]) for r in fresh],
            )

            # ── process ────────────────────────────────────────────────
            batch_outcomes = await self._process_batch(fresh)
            outcomes.extend(batch_outcomes)

            if len(outcomes) >= total_needed:
                break

        return {
            "outcomes": {s: outcomes.count(s) for s in sorted(set(outcomes))},
            "total_scanned": scanned,
            "skipped_by_checkpoint": skipped_by_checkpoint,
            "total_processed": len(outcomes),
        }

    # ---- batch processing -------------------------------------------------

    async def _process_batch(
        self, batch: list[dict[str, Any]],
    ) -> list[str]:
        # Process every item concurrently — each returns (outcome, dedup_entry).
        results = await asyncio.gather(
            *(self._process_one(entry) for entry in batch),
        )

        # Collect dedup updates for a single SQL transaction.
        dedup_entries: list[DedupEntry] = []
        outcomes: list[str] = []
        for outcome, dedup_entry in results:
            outcomes.append(outcome)
            if dedup_entry is not None:
                dedup_entries.append(dedup_entry)

        if dedup_entries:
            self._dedup_store.update_status_batch(dedup_entries)

        return outcomes

    # ---- per-item processing ----------------------------------------------

    async def _process_one(
        self, record: dict[str, Any],
    ) -> tuple[str, DedupEntry | None]:
        """Label, validate, and (on failure) repair a single record.

        Returns:
            ``(outcome, dedup_entry)``.  *dedup_entry* is ``None`` for skips.
        """
        item_id = str(record["id"])
        hash_ = str(record["recipe_card_hash"])

        # Already marked completed in a prior run → skip.
        item = self._checkpoint.item(item_id)
        if item["status"] == "completed":
            return "skipped_completed", None

        raw_text = str(record["raw_text"])
        last_error = ""
        for _ in range(self._retries + 1):
            attempt = await self._checkpoint.mark_in_flight(item_id, hash_)
            parsed = None
            result = None
            try:
                result = await self._client.label(raw_text)
                parsed = self._validator.parse(result.raw_output)
            except OutputValidationError as error:
                last_error = str(error)
                logger.warning(
                    "Labeling validation failed id=%s attempt=%d error=%s",
                    item_id, attempt, last_error,
                )
                if result is not None:
                    try:
                        repaired = await self._client.repair(
                            result.raw_output, last_error,
                        )
                        parsed = self._validator.parse(repaired.raw_output)
                        logger.info("Repair succeeded id=%s attempt=%d", item_id, attempt)
                        result = repaired
                    except Exception as repair_error:
                        logger.warning(
                            "Repair failed id=%s attempt=%d error=%s",
                            item_id, attempt, repair_error,
                        )
                        continue
                else:
                    continue
            except Exception as error:
                last_error = str(error)
                logger.warning(
                    "Labeling attempt failed id=%s attempt=%d error=%s",
                    item_id, attempt, last_error,
                )
                continue

            if parsed is None or result is None:  # pragma: no cover — defensive
                continue

            outcome = "not_a_recipe" if parsed.is_not_a_recipe else "completed"

            await self._checkpoint.mark_completed(item_id, output_line=None)
            logger.info(
                "Labeling %s id=%s attempts=%d",
                outcome, item_id, attempt,
            )
            return outcome, DedupEntry(
                recipe_card_hash=hash_,
                status=HashStatus.ACCEPTED,
                raw_text=raw_text,
                model=result.model,
                output=json.dumps(parsed.normalized_json, ensure_ascii=False),
            )

        # All retries exhausted.
        await self._checkpoint.mark_failed(item_id, last_error)
        logger.error("Labeling exhausted retries id=%s error=%s", item_id, last_error)
        return "failed", DedupEntry(
            recipe_card_hash=hash_, status=HashStatus.REJECTED,
        )
