"""Persistent FIFO work queue backed by an atomic JSON checkpoint file."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..interfaces.work_queue import WorkQueue


class CheckpointQueue(WorkQueue):
    """Persist queue state in an atomic JSON file.

    Items track an attempt counter.  :meth:`dequeue` returns items whose
    attempt count is below *max_attempts*, incrementing the counter each
    time.  Items that reach the limit are silently dropped — the sampler
    will still re-discover them, but :meth:`enqueue` skips known ids so
    they stay dead across runs.
    """

    def __init__(self, path: Path, *, max_attempts: int = 10) -> None:
        self._path = path
        self._items: dict[str, dict[str, Any]] = {}
        self._max_attempts = max_attempts
        self._lock = asyncio.Lock()

    # ---- WorkQueue interface ----------------------------------------------

    async def load(self) -> None:
        if not self._path.exists():
            return
        state = json.loads(self._path.read_text(encoding="utf-8"))
        if state.get("version") != 1:
            raise ValueError(
                f"unsupported checkpoint version: {state.get('version')!r}"
            )
        self._items = state.get("items", {})

    async def enqueue(self, items: list[dict[str, Any]]) -> None:
        """Add *items* with ``attempts=0``.  Known ids are left untouched
        (including items that have exhausted retries — they stay dead)."""
        changed = False
        for record in items:
            item_id = str(record["id"])
            if item_id in self._items:
                continue
            self._items[item_id] = {
                "id": item_id,
                "source_id": str(record.get("source_id", "")),
                "recipe_card_hash": str(record.get("recipe_card_hash", "")),
                "raw_text": str(record.get("raw_text", "")),
                "attempts": 0,
            }
            changed = True
        if changed:
            await self._persist()

    async def dequeue(self, count: int) -> list[dict[str, Any]]:
        """Remove and return up to *count* items whose attempt count is
        below *max_attempts*.  Each returned item's counter is incremented;
        items that have reached the limit are permanently dropped."""
        ready: list[dict[str, Any]] = []
        keys: list[str] = []
        for key, item in self._items.items():
            if int(item.get("attempts", 0)) >= self._max_attempts:
                continue
            ready.append(item)
            keys.append(key)
            if len(ready) >= count:
                break

        for key in keys:
            item = self._items[key]
            item["attempts"] = int(item.get("attempts", 0)) + 1
            if item["attempts"] >= self._max_attempts:
                # Keep the item in _items as a tombstone so enqueue skips it.
                pass
            else:
                del self._items[key]

        if ready:
            await self._persist()
        return ready

    # ---- internal ---------------------------------------------------------

    async def _persist(self) -> None:
        async with self._lock:
            payload = {
                "version": 1,
                "updated_at": self._timestamp(),
                "items": {
                    item_id: {
                        k: v for k, v in item.items()
                        if k != "raw_text"
                    }
                    for item_id, item in self._items.items()
                },
            }
            encoded = (
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
                + "\n"
            )
            self._path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, tmp = tempfile.mkstemp(
                prefix=f".{self._path.name}.", dir=self._path.parent,
            )
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                    handle.write(encoded)
                    handle.flush()
                    os.fsync(handle.fileno())
                try:
                    os.replace(tmp, self._path)
                except PermissionError:
                    backup = tempfile.mktemp(
                        prefix=f".{self._path.name}.bak.", dir=self._path.parent,
                    )
                    os.rename(self._path, backup)
                    try:
                        os.rename(tmp, self._path)
                    except BaseException:
                        os.rename(backup, self._path)
                        raise
                    else:
                        os.unlink(backup)
            finally:
                if os.path.exists(tmp):
                    os.unlink(tmp)

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
