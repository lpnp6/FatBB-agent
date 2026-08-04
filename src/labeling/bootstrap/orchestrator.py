"""Bootstrap orchestrator — discover, deduplicate, label, validate, repair, persist."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from ..interfaces.dedup_store import DedupEntry, DedupStore, HashStatus
from ..interfaces.checkpoint_store import CheckpointStore
from ..interfaces.labeling_client import LabelingClient
from ..interfaces.orchestrator import Orchestrator
from ..sampling.sampler import Sampler
from ..utils.validator import OutputValidationError, OutputValidator


logger = logging.getLogger(__name__)


class BootstrapOrchestrator(Orchestrator):
    """Complete bootstrap pipeline: discover → label → persist.

    Owns the :class:`Sampler` and drives the full lifecycle:

    1. Stream batches from :meth:`Sampler.sample`.
    2. Label / validate / repair concurrently within each batch.
    3. On success: write ACCEPTED + provenance to dedup store.
    4. Checkpoint state provides resume behavior.
    """

    def __init__(
        self,
        *,
        client: LabelingClient,
        dedup_store: DedupStore,
        sampler: Sampler,
        checkpoint: CheckpointStore,
        validator: OutputValidator | None = None,
        retries: int = 2,
    ) -> None:
        self._client = client
        self._dedup_store = dedup_store
        self._sampler = sampler
        self._checkpoint = checkpoint
        self._validator = validator or OutputValidator()
        self._retries = retries


    # ---- Orchestrator interface -------------------------------------------

    async def run(
        self,
        root: Path | str,
        target: int,
        *,
        holdout: int = 0,
        glob: str = "**/*.md",
    ) -> dict[str, Any]:
        await self._checkpoint.load()
        # Clean up IN_FLIGHT items left by a previous crash so they can be
        # re-discovered and re-processed.
        self._dedup_store.expire_stale(timeout_minutes=0)

        scanned = 0
        outcomes: list[str] = []

        async for batch in self._sampler.sample(root, glob=glob):
            scanned += len(batch)

            ready = [
                {
                    "source_id": sid,
                    "recipe_card_hash": h,
                    "raw_text": text,
                }
                for sid, h, text in batch
            ]

            # Pre-register IN_FLIGHT in dedup + checkpoint.
            self._dedup_store.register_batch([
                DedupEntry(
                    recipe_card_hash=str(r["recipe_card_hash"]),
                    status=HashStatus.IN_FLIGHT,
                    source_id=str(r["source_id"]),
                )
                for r in ready
            ])
            await self._checkpoint.mark_in_flight_batch([
                (str(r["source_id"]), str(r["recipe_card_hash"]))
                for r in ready
            ])

            results = await asyncio.gather(
                *(self._process_one(item) for item in ready),
            )

            dedup_entries: list[DedupEntry] = []
            for outcome, entry in results:
                outcomes.append(outcome)
                if entry is not None:
                    dedup_entries.append(entry)

            if dedup_entries:
                self._dedup_store.register_batch(dedup_entries)
                await self._checkpoint.mark_completed_batch(
                    [str(e.source_id) for e in dedup_entries],
                )

            if len(outcomes) >= target + holdout:
                break

        return {
            "outcomes": {s: outcomes.count(s) for s in sorted(set(outcomes))},
            "total_scanned": scanned,
            "total_processed": len(outcomes),
        }

    # ---- per-item processing ----------------------------------------------

    async def _process_one(
        self, item: dict[str, Any],
    ) -> tuple[str, DedupEntry | None]:
        """Label, validate, repair.  Returns staged dedup entry for batch flush.

        On failure after all retries, the checkpoint is marked rejected.
        """
        item_id = str(item["source_id"])
        hash_ = str(item["recipe_card_hash"])
        raw_text = str(item["raw_text"])
        source_id = item_id

        last_error = ""
        for attempt_num in range(self._retries + 1):
            parsed = None
            result = None
            try:
                result = await self._client.label(raw_text)
                parsed = self._validator.parse(result.raw_output)
            except OutputValidationError as error:
                last_error = str(error)
                logger.warning(
                    "Labeling validation failed id=%s attempt=%d error=%s",
                    item_id, attempt_num + 1, last_error,
                )
                if result is not None:
                    try:
                        repaired = await self._client.repair(
                            result.raw_output, last_error,
                        )
                        parsed = self._validator.parse(repaired.raw_output)
                        logger.info("Repair succeeded id=%s attempt=%d", item_id, attempt_num + 1)
                        result = repaired
                    except Exception as repair_error:
                        logger.warning(
                            "Repair failed id=%s attempt=%d error=%s",
                            item_id, attempt_num + 1, repair_error,
                        )
                        continue
                else:
                    continue
            except Exception as error:
                last_error = str(error)
                logger.warning(
                    "Labeling attempt failed id=%s attempt=%d error=%s",
                    item_id, attempt_num + 1, last_error,
                )
                continue

            if parsed is None or result is None:  # pragma: no cover — defensive
                continue

            outcome = "not_a_recipe" if parsed.is_not_a_recipe else "completed"
            logger.info(
                "Labeling %s id=%s attempts=%d",
                outcome, item_id, attempt_num + 1,
            )
            return outcome, DedupEntry(
                recipe_card_hash=hash_,
                status=HashStatus.ACCEPTED,
                source_id=source_id,
                raw_text=raw_text,
                model=result.model,
                output=json.dumps(parsed.normalized_json, ensure_ascii=False),
            )

        # All retries exhausted. The sampler skips rejected items on later runs.
        logger.error("Labeling exhausted retries id=%s error=%s", item_id, last_error)
        await self._checkpoint.mark_rejected(item_id, last_error)
        return "failed", None
