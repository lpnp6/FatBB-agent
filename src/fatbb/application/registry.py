"""Resolve knowledge-base type keys through a source-controlled catalog."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
import tomllib

from fatbb.domain.ports import KnowledgeBaseAdapter, SourceImporter


@dataclass(frozen=True)
class CapabilityDefinition:
    """One configured knowledge-base module factory."""

    key: str
    factory: str
    database_type: str | None = None


class CapabilityRegistry:
    """Resolve persisted capability identifiers to concrete adapters.

    Knowledge bases store small, stable strings such as ``"bm25"`` and
    ``"file_path"`` in :class:`KnowledgeBaseConfig`; they must not depend on
    Python implementation classes. This registry is the composition boundary
    that maps those strings to the adapters available in this deployment.

    Adapter mappings live in ``kb.toml`` and are versioned with the
    source code. A knowledge base persists only a stable key; when needed, the
    registry builds the adapter configured for that key.
    """

    def __init__(self, config_path: Path):
        payload = tomllib.loads(config_path.read_text(encoding="utf-8"))
        self._database_types = self._keys(payload, "databases")
        self._knowledge_bases = self._definitions(payload, "knowledge_bases")
        self._importers = self._definitions(payload, "importers")

    def knowledge_base(self, retrieval_type: str, database_type: str) -> KnowledgeBaseAdapter:
        """Build the knowledge-base adapter configured for stored type keys."""
        try:
            definition = self._knowledge_bases[retrieval_type]
        except KeyError as error:
            raise ValueError(f"Unsupported retrieval type: {retrieval_type}") from error
        if definition.database_type != database_type:
            raise ValueError(
                f"Retrieval type {retrieval_type!r} does not support database type {database_type!r}."
            )
        if database_type not in self._database_types:
            raise ValueError(f"Unsupported database type: {database_type}")
        return self._build(definition, KnowledgeBaseAdapter)

    def importer(self, source_type: str) -> SourceImporter:
        """Build the importer configured for a stored type key."""
        try:
            return self._build(self._importers[source_type], SourceImporter)
        except KeyError as error:
            raise ValueError(f"Unsupported source type: {source_type}") from error

    @staticmethod
    def _keys(payload: dict[str, object], group: str) -> tuple[str, ...]:
        raw_group = payload.get(group)
        if not isinstance(raw_group, dict):
            raise ValueError(f"Capability config requires a [{group}] section.")
        keys: list[str] = []
        for key, raw_value in raw_group.items():
            if not isinstance(key, str) or not isinstance(raw_value, dict):
                raise ValueError(f"Invalid capability definition in {group}.")
            keys.append(key)
        return tuple(keys)

    @staticmethod
    def _definitions(payload: dict[str, object], group: str) -> dict[str, CapabilityDefinition]:
        """Parse one TOML capability group without importing any adapter code.

        The result contains only stable keys, display labels, and factory-path
        strings. This startup-time validation is deliberately separate from
        :meth:`_build`, so unused capabilities are never imported merely
        because they appear in the catalog.
        """
        raw_group = payload.get(group)
        if not isinstance(raw_group, dict):
            raise ValueError(f"Capability config requires a [{group}] section.")
        definitions: dict[str, CapabilityDefinition] = {}
        for key, raw_value in raw_group.items():
            if not isinstance(key, str) or not isinstance(raw_value, dict):
                raise ValueError(f"Invalid capability definition in {group}.")
            factory = raw_value.get("factory")
            database_type = raw_value.get("database_type")
            if not isinstance(factory, str):
                raise ValueError(f"Capability {key!r} requires a factory string.")
            if group == "knowledge_bases" and not isinstance(database_type, str):
                raise ValueError(f"Knowledge-base capability {key!r} requires a database_type string.")
            definitions[key] = CapabilityDefinition(
                key=key, factory=factory,
                database_type=database_type if isinstance(database_type, str) else None,
            )
        return definitions

    @staticmethod
    def _build(definition: CapabilityDefinition, expected_type):
        """Lazily import and instantiate the adapter named by one definition.

        This is called only after a knowledge base requests its saved type key.
        For example, ``bm25`` imports ``PostgresBm25KnowledgeBase`` here, rather than
        while parsing ``kb.toml``. The method also verifies the small
        runtime method surface required by the selected domain port.
        """
        try:
            # Factory paths use ``package.module:ClassName`` syntax so the TOML
            # file can be reviewed and changed independently of app.py.
            module_name, class_name = definition.factory.split(":", maxsplit=1)
            factory = getattr(import_module(module_name), class_name)
            # The adapter itself is short-lived and receives connection details
            # later through its indexer()/retriever() method.
            adapter = factory()
        except (ImportError, AttributeError, TypeError, ValueError) as error:
            raise RuntimeError(
                f"Unable to build capability {definition.key!r} from {definition.factory!r}."
            ) from error
        # Protocols are structural and cannot be used as a runtime isinstance
        # check; verify the minimal method surface expected by each caller.
        required = (
            ("check_connection", "indexer", "retriever")
            if expected_type is KnowledgeBaseAdapter
            else ("load",)
        )
        if not all(callable(getattr(adapter, name, None)) for name in required):
            raise RuntimeError(f"Capability {definition.key!r} does not implement its required adapter port.")
        return adapter
