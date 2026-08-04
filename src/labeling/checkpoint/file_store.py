"""Atomic, resume-safe checkpoint storage backed by a local JSON file."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..interfaces.checkpoint_store import CheckpointStore, ItemStatus


class FileCheckpointStore(CheckpointStore):
    """Persist per-manifest-item progress in a JSON file after every transition.

    Atomic writes via tempfile + os.replace with a Windows ``PermissionError``
    fallback so the checkpoint file is never left in a half-written state.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._state: dict[str, Any] = {}
        self._lock = asyncio.Lock()

    # ---- lifecycle -----------------------------------------------------------

    async def load(self) -> None:
        if self._path.exists():
            raw: object = json.loads(self._path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError(f"checkpoint is not a JSON object: {type(raw)}")
            self._state = raw
            if self._state.get("version") != 1:
                raise ValueError(f"unsupported checkpoint version: {self._state.get('version')!r}")
        else:
            self._state = {
                "version": 1,
                "created_at": self._timestamp(),
                "items": {},
            }
            await self._persist()

    async def ensure_items(self, item_ids: list[str]) -> None:
        items: dict[str, Any] = self._state["items"]  # type: ignore[assignment]
        changed = False
        for item_id in item_ids:
            if item_id not in items:
                items[item_id] = {
                    "status": ItemStatus.PENDING.value,
                    "attempts": 0,
                    "error": None,
                }
                changed = True
        if changed:
            await self._persist()

    # ---- accessors -----------------------------------------------------------

    def get_status(self, item_id: str) -> ItemStatus:
        return ItemStatus(str(self._raw_item(item_id)["status"]))

    def get_attempt(self, item_id: str) -> int:
        return int(self._raw_item(item_id)["attempts"])

    # ---- staleness -----------------------------------------------------------

    async def expire_stale(self) -> int:
        """Remove all IN_FLIGHT items — they will be re-created as PENDING
        by the next :meth:`ensure_items` call."""
        items: dict[str, dict[str, Any]] = self._state["items"]  # type: ignore[assignment]
        stale = [
            iid for iid, item in items.items()
            if item.get("status") == ItemStatus.IN_FLIGHT.value
        ]
        for iid in stale:
            del items[iid]
        if stale:
            await self._persist()
        return len(stale)

    # ---- state transitions ---------------------------------------------------

    async def mark_in_flight(self, item_id: str, recipe_card_hash: str) -> int:
        item = self._raw_item(item_id)
        item.update({
            "status": ItemStatus.IN_FLIGHT.value,
            "attempts": int(item["attempts"]) + 1,
            "recipe_card_hash": recipe_card_hash,
            "error": None,
        })
        await self._persist()
        return int(item["attempts"])

    async def mark_completed(self, item_id: str) -> None:
        item = self._raw_item(item_id)
        item.update({"status": ItemStatus.COMPLETED.value, "error": None})
        await self._persist()

    async def mark_in_flight_batch(
        self, items: list[tuple[str, str]],
    ) -> None:
        """Mark multiple items IN_FLIGHT in one persist."""
        if not items:
            return
        for item_id, recipe_card_hash in items:
            item = self._raw_item(item_id)
            item.update({
                "status": ItemStatus.IN_FLIGHT.value,
                "attempts": int(item["attempts"]) + 1,
                "recipe_card_hash": recipe_card_hash,
                "error": None,
            })
        await self._persist()

    async def mark_completed_batch(self, item_ids: list[str]) -> None:
        """Mark multiple items COMPLETED in one persist."""
        if not item_ids:
            return
        for item_id in item_ids:
            item = self._raw_item(item_id)
            item.update({
                "status": ItemStatus.COMPLETED.value,
                "error": None,
            })
        await self._persist()

    async def mark_rejected(self, item_id: str, error: str) -> None:
        self._raw_item(item_id).update({"status": ItemStatus.REJECTED.value, "error": error})
        await self._persist()

    async def mark_pending(self, item_id: str) -> None:
        self._raw_item(item_id).update({"status": ItemStatus.PENDING.value})
        await self._persist()

    # ---- monitoring ----------------------------------------------------------

    def get_stats(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        items: dict[str, dict[str, Any]] = self._state["items"]  # type: ignore[assignment]
        for item in items.values():
            status = str(item["status"])
            counts[status] = counts.get(status, 0) + 1
        return counts

    # ---- internal ------------------------------------------------------------

    def _raw_item(self, item_id: str) -> dict[str, Any]:
        items: dict[str, dict[str, Any]] = self._state["items"]  # type: ignore[assignment]
        return items[item_id]

    async def _persist(self) -> None:
        async with self._lock:
            self._state["updated_at"] = self._timestamp()
            self._path.parent.mkdir(parents=True, exist_ok=True)
            payload = json.dumps(self._state, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{self._path.name}.", dir=self._path.parent,
            )
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
                try:
                    os.replace(temporary_name, self._path)
                except PermissionError:
                    # Windows: os.replace may fail with ERROR_ACCESS_DENIED when
                    # the target exists.  Rename the old file out of the way,
                    # move the new file into place, then remove the backup —
                    # the old checkpoint is never deleted until the new one is
                    # safely in place.
                    backup = tempfile.mktemp(
                        prefix=f".{self._path.name}.bak.", dir=self._path.parent,
                    )
                    os.rename(self._path, backup)
                    try:
                        os.rename(temporary_name, self._path)
                    except BaseException:
                        os.rename(backup, self._path)
                        raise
                    else:
                        os.unlink(backup)
            finally:
                if os.path.exists(temporary_name):
                    os.unlink(temporary_name)

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
