"""Atomic, resume-safe checkpoint storage for bootstrap labeling."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class CheckpointManager:
    """Persist per-manifest-item progress in a JSON file after every transition."""

    def __init__(self, path: Path, *, manifest_path: Path, output_path: Path) -> None:
        self._path = path
        self._manifest_path = str(manifest_path)
        self._output_path = str(output_path)
        self._state: dict[str, Any] = {}
        self._lock = asyncio.Lock()

    async def load(self) -> dict[str, Any]:
        if self._path.exists():
            self._state = json.loads(self._path.read_text(encoding="utf-8"))
            if self._state.get("version") != 1:
                raise ValueError(f"unsupported checkpoint version: {self._state.get('version')!r}")
        else:
            self._state = {
                "version": 1,
                "manifest_path": self._manifest_path,
                "output_path": self._output_path,
                "created_at": self._timestamp(),
                "items": {},
            }
            await self._persist()
        return self._state

    async def ensure_items(self, item_ids: list[str]) -> None:
        items = self._state["items"]
        changed = False
        for item_id in item_ids:
            if item_id not in items:
                items[item_id] = {"status": "pending", "attempts": 0, "output_line": None, "error": None}
                changed = True
        if changed:
            await self._persist()

    def item(self, item_id: str) -> dict[str, Any]:
        return self._state["items"][item_id]

    async def mark_in_flight(self, item_id: str, recipe_card_hash: str) -> int:
        item = self.item(item_id)
        item.update({
            "status": "in_flight",
            "attempts": int(item["attempts"]) + 1,
            "recipe_card_hash": recipe_card_hash,
            "error": None,
        })
        await self._persist()
        return int(item["attempts"])

    async def mark_completed(self, item_id: str, *, output_line: int | None) -> None:
        item = self.item(item_id)
        item.update({"status": "completed", "output_line": output_line, "error": None})
        await self._persist()

    async def mark_failed(self, item_id: str, error: str) -> None:
        self.item(item_id).update({"status": "failed", "error": error})
        await self._persist()

    async def _persist(self) -> None:
        async with self._lock:
            self._state["updated_at"] = self._timestamp()
            self._path.parent.mkdir(parents=True, exist_ok=True)
            payload = json.dumps(self._state, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
            descriptor, temporary_name = tempfile.mkstemp(prefix=f".{self._path.name}.", dir=self._path.parent)
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
                try:
                    os.replace(temporary_name, self._path)
                except PermissionError:
                    # Windows: os.replace may fail with ERROR_ACCESS_DENIED when the
                    # target exists.  Rename the old file out of the way first, move
                    # the new file into place, then remove the backup — the old
                    # checkpoint is never deleted until the new one is safely in place.
                    backup = tempfile.mktemp(
                        prefix=f".{self._path.name}.bak.", dir=self._path.parent
                    )
                    os.rename(self._path, backup)
                    try:
                        os.rename(temporary_name, self._path)
                    except BaseException:
                        # Restore the old checkpoint on failure
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
