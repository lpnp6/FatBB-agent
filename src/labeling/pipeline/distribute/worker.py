"""Worker that consumes tasks from a WorkQueue and labels them via a LabelingClient."""

from __future__ import annotations

import json
import logging
from typing import Any

from ...interfaces.dedup_store import DedupEntry, HashStatus
from ...interfaces.labeling_client import LabelingClient, TransientError
from ...interfaces.work_queue import WorkQueue
from ...utils.validator import OutputValidationError, OutputValidator

logger = logging.getLogger(__name__)


class Worker:
    """Dequeue tasks, call the model, validate, and submit results back.

    Accepts only abstract interfaces — queue backend and model provider are
    injected at construction time.
    """

    def __init__(
        self,
        *,
        queue: WorkQueue,
        client: LabelingClient,
        validator: OutputValidator | None = None,
        repair: bool = True,
    ) -> None:
        self._queue = queue
        self._client = client
        self._validator = validator or OutputValidator()
        self._repair = repair

    async def run(self, *, count: int = 1) -> None:
        """Block in a dequeue→label→submit loop."""
        await self._queue.load()
        while True:
            tasks = await self._queue.dequeue(count)
            if not tasks:
                continue
            outcomes: list[tuple[dict[str, Any], DedupEntry]] = []
            failures: list[tuple[dict[str, Any], Exception]] = []
            for task in tasks:
                try:
                    outcomes.append((task, await self._process_one(task)))
                except Exception as exc:
                    logger.warning(
                        "Task failed source_id=%s error=%s",
                        task.get("source_id"), exc,
                    )
                    failures.append((task, exc))
            if outcomes:
                await self._queue.submit_results(outcomes)
            if failures:
                await self._queue.submit_retries(failures)

    async def _process_one(self, task: dict[str, Any]) -> DedupEntry:
        raw_text = str(task["raw_text"])
        result = await self._client.label(raw_text)

        try:
            parsed = self._validator.parse(result.raw_output)
        except OutputValidationError as exc:
            if self._repair:
                result = await self._client.repair(result.raw_output, str(exc))
                parsed = self._validator.parse(result.raw_output)
            else:
                raise

        return DedupEntry(
            recipe_card_hash=str(task["recipe_card_hash"]),
            status=HashStatus.ACCEPTED,
            source_id=str(task["source_id"]),
            raw_text=raw_text,
            model=result.model,
            output=json.dumps(parsed.normalized_json, ensure_ascii=False),
        )
