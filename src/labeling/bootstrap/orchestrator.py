"""Resume-safe execution of a bootstrap labeling manifest."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any

from .checkpoint import CheckpointManager
from ..interfaces.dedup_store import DedupStore, HashStatus
from ..interfaces.labeling_client import LabelingClient
from .validator import OutputValidationError, OutputValidator


logger = logging.getLogger(__name__)


class JsonlTrainingWriter:
    """Append durable, idempotent records and recover line numbers on restart."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lines_by_id: dict[str, int] = {}
        if path.exists():
            for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
                if line.strip():
                    record = json.loads(line)
                    self._lines_by_id[str(record["id"])] = line_number

    def existing_line(self, item_id: str) -> int | None:
        return self._lines_by_id.get(item_id)

    def append(self, record: dict[str, Any]) -> int:
        item_id = str(record["id"])
        existing = self.existing_line(item_id)
        if existing is not None:
            return existing
        self._path.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        line_number = len(self._lines_by_id) + 1
        self._lines_by_id[item_id] = line_number
        return line_number


class LabelingPipeline:
    """Process a manifest concurrently while preserving per-item durability."""

    def __init__(
        self,
        *,
        client: LabelingClient,
        dedup_store: DedupStore,
        checkpoint: CheckpointManager,
        training_writer: JsonlTrainingWriter,
        validator: OutputValidator | None = None,
        retries: int = 2,
    ) -> None:
        self._client = client
        self._dedup_store = dedup_store
        self._checkpoint = checkpoint
        self._writer = training_writer
        self._validator = validator or OutputValidator()
        self._retries = retries

    async def run(self, manifest: list[dict[str, Any]]) -> dict[str, int]:
        await self._checkpoint.load()
        await self._checkpoint.ensure_items([str(entry["id"]) for entry in manifest])
        results = await asyncio.gather(*(self._process(entry) for entry in manifest))
        return {status: results.count(status) for status in sorted(set(results))}

    async def _process(self, entry: dict[str, Any]) -> str:
        item_id = str(entry["id"])
        item = self._checkpoint.item(item_id)
        if item["status"] == "completed":
            return "skipped_completed"

        existing_line = self._writer.existing_line(item_id)
        if existing_line is not None:
            self._dedup_store.update_status(str(entry["recipe_card_hash"]), HashStatus.ACCEPTED)
            await self._checkpoint.mark_completed(item_id, output_line=existing_line)
            return "recovered_output"

        path = Path(str(entry["path"]))
        last_error = ""
        for _ in range(self._retries + 1):
            attempt = await self._checkpoint.mark_in_flight(item_id, str(entry["recipe_card_hash"]))
            parsed = None
            result = None
            try:
                markdown = path.read_text(encoding="utf-8", errors="replace")
                result = await self._client.label(markdown)
                parsed = self._validator.parse(result.raw_output)
            except OutputValidationError as error:
                last_error = str(error)
                logger.warning("Labeling validation failed id=%s attempt=%d error=%s", item_id, attempt, last_error)
                if result is not None:
                    try:
                        repaired = await self._client.repair(result.raw_output, last_error)
                        parsed = self._validator.parse(repaired.raw_output)
                        logger.info("Repair succeeded id=%s attempt=%d", item_id, attempt)
                        result = repaired
                    except Exception as repair_error:
                        logger.warning("Repair failed id=%s attempt=%d error=%s", item_id, attempt, repair_error)
                        continue
                else:
                    continue
            except Exception as error:
                last_error = str(error)
                logger.warning("Labeling attempt failed id=%s attempt=%d error=%s", item_id, attempt, last_error)
                continue

            if parsed is None or result is None:  # pragma: no cover — defensive
                continue

            if parsed.is_not_a_recipe:
                line_number = self._writer.append({
                    "id": item_id,
                    "source_path": str(path),
                    "recipe_card_hash": str(entry["recipe_card_hash"]),
                    "model": result.model,
                    "input": markdown,
                    "output": parsed.normalized_json,
                    "is_not_a_recipe": True,
                })
                self._dedup_store.update_status(str(entry["recipe_card_hash"]), HashStatus.ACCEPTED)
                await self._checkpoint.mark_completed(item_id, output_line=line_number)
                logger.info("Labeling completed as not_a_recipe id=%s output_line=%d attempts=%d", item_id, line_number, attempt)
                return "not_a_recipe"
            line_number = self._writer.append({
                "id": item_id,
                "source_path": str(path),
                "recipe_card_hash": str(entry["recipe_card_hash"]),
                "model": result.model,
                "input": markdown,
                "output": parsed.normalized_json,
            })
            self._dedup_store.update_status(str(entry["recipe_card_hash"]), HashStatus.ACCEPTED)
            await self._checkpoint.mark_completed(item_id, output_line=line_number)
            logger.info("Labeling completed id=%s output_line=%d attempts=%d", item_id, line_number, attempt)
            return "completed"

        self._dedup_store.update_status(str(entry["recipe_card_hash"]), HashStatus.REJECTED)
        await self._checkpoint.mark_failed(item_id, last_error)
        logger.error("Labeling exhausted retries id=%s error=%s", item_id, last_error)
        return "failed"
