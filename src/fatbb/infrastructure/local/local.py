"""Private local persistence for the knowledge-base catalog."""

from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import tempfile
from typing import TypedDict, cast

from fatbb.domain.knowledge_base import KnowledgeBase, KnowledgeBaseConfig


class _LocalCatalog(TypedDict):
    """Validated on-disk shape of the local knowledge-base catalog."""

    version: int
    knowledge_bases: list[dict[str, object]]


class Local:
    """Own local CLI persistence, beginning with the knowledge-base catalog.

    A knowledge base's PostgreSQL URL can include credentials, so its catalog
    is created with owner-only permissions on POSIX systems. Future local
    concerns, such as conversation history, belong here as separate files and
    methods on the same local-storage facade.
    """

    def __init__(self, path: Path | None = None):
        self._path = path or Path.home() / ".fatbb" / "knowledge_bases.json"

    def list_knowledge_bases(self) -> list[KnowledgeBase]:
        payload = self._read()
        return [self._to_model(item) for item in payload["knowledge_bases"]]

    def create_knowledge_base(self, knowledge_base: KnowledgeBase) -> None:
        payload = self._read()
        if any(item["name"] == knowledge_base.name for item in payload["knowledge_bases"]):
            raise ValueError(f'A knowledge base named "{knowledge_base.name}" already exists.')
        payload["knowledge_bases"].append(asdict(knowledge_base))
        self._write(payload)

    def _read(self) -> _LocalCatalog:
        """Load and validate the local catalog before exposing typed entries.

        Missing configuration is treated as a first-run empty catalog. Existing
        JSON is validated at this boundary so the rest of ``Local`` can safely
        iterate its knowledge-base entries without handling untyped JSON.
        """
        if not self._path.exists():
            return {"version": 1, "knowledge_bases": []}
        try:
            payload: object = json.loads(self._path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid knowledge-base configuration: {self._path}") from error
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid knowledge-base configuration: {self._path}")
        version = payload.get("version")
        entries = payload.get("knowledge_bases")
        if not isinstance(version, int) or not isinstance(entries, list):
            raise ValueError(f"Invalid knowledge-base configuration: {self._path}")
        if not all(isinstance(entry, dict) for entry in entries):
            raise ValueError(f"Invalid knowledge-base configuration: {self._path}")
        # The JSON value is unknown until validated. Return a precise type so
        # callers know ``knowledge_bases`` is iterable and contains mappings.
        return _LocalCatalog(
            version=version,
            knowledge_bases=cast(list[dict[str, object]], entries),
        )

    def _write(self, payload: _LocalCatalog) -> None:
        """Atomically replace the local catalog and restrict its permissions.

        The temporary file is created beside the destination, then renamed with
        ``os.replace`` so a crash cannot leave a partially-written catalog.
        Both the directory and final file are owner-only on POSIX systems,
        because a PostgreSQL URL may contain credentials.
        """
        self._path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        try:
            os.chmod(self._path.parent, 0o700)
        except OSError:
            pass
        descriptor, temporary_name = tempfile.mkstemp(
            prefix="knowledge_bases.", suffix=".tmp", dir=self._path.parent
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as file:
                json.dump(payload, file, indent=2, sort_keys=True)
                file.write("\n")
            os.chmod(temporary_path, 0o600)
            os.replace(temporary_path, self._path)
            os.chmod(self._path, 0o600)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()

    @staticmethod
    def _to_model(value: object) -> KnowledgeBase:
        """Convert one JSON catalog entry back into an immutable domain model.

        Local storage is untyped JSON, while application code expects a valid
        ``KnowledgeBase`` and ``KnowledgeBaseConfig``. Validate the outer entry
        and nested configuration here so corrupted or manually edited catalog
        data fails at the storage boundary rather than later during retrieval.
        """
        if not isinstance(value, dict):
            raise ValueError("Invalid knowledge-base entry.")
        config = value.get("config")
        if not isinstance(config, dict):
            raise ValueError("Invalid knowledge-base configuration entry.")
        try:
            return KnowledgeBase(
                id=str(value["id"]),
                name=str(value["name"]),
                config=KnowledgeBaseConfig(
                    retrieval_type=str(config["retrieval_type"]),
                    database_type=str(config["database_type"]),
                    database_url=str(config["database_url"]),
                    source_type=str(config["source_type"]),
                ),
                source_path=str(value["source_path"]),
            )
        except KeyError as error:
            raise ValueError("Invalid knowledge-base entry.") from error
