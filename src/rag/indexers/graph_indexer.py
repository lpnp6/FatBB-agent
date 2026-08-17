"""Config-driven knowledge-graph construction.

``GraphIndexer`` turns a document's structured payload into candidate
``GraphNode``/``GraphEdge`` values using a schema loaded from a config file,
resolves them against the store, fuses duplicates, and upserts. Node labels,
relationship types, and property fields all come from the config, so the
indexer stays independent of any business domain.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Sequence
from pathlib import Path

from tqdm import tqdm

from ..interfaces.indexer import Indexer
from ..interfaces.stores import GraphStore
from ..models.common import SourceRef
from ..models.document import Document
from ..models.graph import (
    GraphEdge,
    GraphNode,
    merge_properties,
    merge_provenance,
    slug,
)

logger = logging.getLogger(__name__)


def load_schema(path: str | Path) -> dict:
    """Load a JSON graph-schema config from ``path``."""
    with Path(path).open(encoding="utf-8") as file:
        return json.load(file)


def _read(obj: object, path: str) -> object:
    """Dot-path lookup returning ``None`` when any segment is missing."""
    for part in path.split("."):
        if not isinstance(obj, dict) or part not in obj:
            return None
        obj = obj[part]
    return obj


def _is_stub_name(name: str) -> bool:
    """A node created only to satisfy a relation endpoint carries its slug as name."""
    return name == slug(name)


def _node_id(label: str, name: str) -> str:
    """Label-namespaced canonical id (``Dish:kung-pao-chicken``) so distinct
    node labels can share a slug without colliding on the unique ``id``."""
    return f"{label}:{slug(name)}"


def _merge_nodes(canonical: GraphNode, candidate: GraphNode) -> GraphNode:
    """Fold ``candidate`` into ``canonical``, upgrading stub names and uniting aliases."""
    name = candidate.name if _is_stub_name(canonical.name) else canonical.name
    names = [*canonical.aliases]
    if candidate.name != canonical.name:
        names.append(candidate.name)
    names.extend(candidate.aliases)
    return GraphNode(
        id=canonical.id,
        label=canonical.label,
        name=name,
        aliases=tuple(dict.fromkeys(names)),
        properties=merge_properties(canonical.properties, candidate.properties),
        provenance=merge_provenance(canonical.provenance, candidate.provenance),
        source=canonical.source or candidate.source,
    )


class GraphIndexer(Indexer):
    """Map documents through a schema config and replace their records in a store."""

    def __init__(self, store: GraphStore, schema: dict):
        self._store = store
        self._schema = schema

    def upsert_documents(
        self, documents: Sequence[Document], *,
        on_progress: Callable[[str, int, int], None] | None = None,
    ) -> None:
        """Map each document to nodes/edges, resolve, fuse, and upsert."""
        total = len(documents)
        logger.info("Upserting documents into knowledge graph", extra={"document_count": total})
        for idx, document in enumerate(
            tqdm(documents, desc="Indexing graph", unit="document", disable=on_progress is not None)
        ):
            if on_progress is not None:
                on_progress("Indexing graph", idx + 1, total)
            mapped = self._map(document)
            if mapped is None:
                continue
            nodes, edges = mapped
            self._resolve_and_upsert(nodes, edges)
        logger.info("Completed graph upsert", extra={"document_count": total})

    def delete_documents(self, document_ids: Sequence[str]) -> None:
        raise NotImplementedError(
            "graph deletion needs provenance tracking (nodes/edges are shared across documents); "
            "not supported in v1"
        )

    def _map(self, document: Document) -> tuple[list[GraphNode], list[GraphEdge]] | None:
        """Map one document's payload to candidate nodes and edges (or ``None`` to skip)."""
        payload = document.metadata.get(self._schema.get("metadata_key", "extraction"))
        if not isinstance(payload, dict):
            return None
        source = document.source
        nodes = self._build_nodes(payload, source)
        edges: list[GraphEdge] = []
        for spec in self._schema.get("edges", []):
            if "relation_map" in spec:
                stub_nodes, stub_edges = self._relation_edges(payload, spec, source)
                nodes.extend(stub_nodes)
                edges.extend(stub_edges)
            else:
                edges.extend(self._object_edges(payload, spec, source))
        return nodes, edges

    def _build_nodes(self, payload: dict, source: SourceRef) -> list[GraphNode]:
        nodes: list[GraphNode] = []
        for label, spec in self._schema.get("nodes", {}).items():
            for node_id, item in self._resolve_nodes(payload, spec, label):
                aliases = item.get(spec["aliases"]) if "aliases" in spec else None
                properties = self._properties(item, spec.get("properties", []))
                nodes.append(GraphNode(
                    node_id, label, item[spec["name"]],
                    tuple(aliases) if aliases else (),
                    properties,
                    source,
                    self._provenance(properties, source),
                ))
        return nodes

    def _resolve_nodes(
        self, payload: dict, spec: dict, label: str,
    ) -> list[tuple[str, dict]]:
        """Yield ``(namespaced_id, item)`` for every item a node spec resolves to."""
        source = _read(payload, spec["source"])
        if source is None:
            return []
        items = source if isinstance(source, list) else [source]
        resolved: list[tuple[str, dict]] = []
        for item in items:
            if isinstance(item, dict) and item.get(spec["name"]):
                resolved.append((_node_id(label, item[spec["name"]]), item))
        return resolved

    def _object_edges(self, payload: dict, spec: dict, source: SourceRef) -> list[GraphEdge]:
        """Emit edges between two node specs (e.g. ``Dish`` → each ``Ingredient``)."""
        from_spec = self._schema["nodes"][spec["from_node"]]
        to_spec = self._schema["nodes"][spec["to_node"]]
        from_ids = [
            node_id for node_id, _ in self._resolve_nodes(payload, from_spec, spec["from_node"])
        ]
        edges: list[GraphEdge] = []
        for node_id, item in self._resolve_nodes(payload, to_spec, spec["to_node"]):
            properties = self._properties(item, spec.get("properties", []))
            for from_id in from_ids:
                edges.append(GraphEdge(
                    from_id, node_id, spec["relation"],
                    properties,
                    source,
                    self._provenance(properties, source),
                ))
        return edges

    def _relation_edges(
        self, payload: dict, spec: dict, source: SourceRef,
    ) -> tuple[list[GraphNode], list[GraphEdge]]:
        """Emit stub nodes and edges for slug-endpoint relations (e.g. ``variant_of``)."""
        nodes: list[GraphNode] = []
        edges: list[GraphEdge] = []
        rows = _read(payload, spec["source"]) or []
        if not isinstance(rows, list):
            rows = [rows]
        for row in rows:
            if not isinstance(row, dict):
                continue
            relation = spec["relation_map"].get(row.get(spec["relation_field"]))
            from_ref = row.get(spec["from_field"])
            to_ref = row.get(spec["to_field"])
            if relation is None or not from_ref or not to_ref:
                continue
            from_id = _node_id(spec["label"], from_ref)
            to_id = _node_id(spec["label"], to_ref)
            nodes.append(GraphNode(from_id, spec["label"], from_ref))
            nodes.append(GraphNode(to_id, spec["label"], to_ref))
            properties = self._properties(row, spec.get("properties", []))
            edges.append(GraphEdge(
                from_id, to_id, relation,
                properties,
                source,
                self._provenance(properties, source),
            ))
        return nodes, edges

    @staticmethod
    def _properties(item: dict, fields: list[str]) -> dict:
        return {field: item[field] for field in fields if item.get(field) is not None}

    @staticmethod
    def _provenance(properties: dict, source: SourceRef | None) -> dict:
        """Tag each property value with the id of the source that contributed it."""
        if not properties:
            return {}
        source_id = source.document_id if source is not None else None
        return {key: {source_id: value} for key, value in properties.items()}

    def _resolve_and_upsert(self, candidates: list[GraphNode], edges: list[GraphEdge]) -> None:
        existing = self._store.resolve_entities(candidates)
        canonical: dict[str, GraphNode] = {}
        remap: dict[str, str] = {}
        for candidate, matched in zip(candidates, existing):
            cid = matched.id if matched is not None else candidate.id
            remap[candidate.id] = cid
            accumulated = canonical.get(cid)
            if accumulated is None:
                canonical[cid] = _merge_nodes(matched, candidate) if matched is not None else candidate
            else:
                canonical[cid] = _merge_nodes(accumulated, candidate)
        resolved_edges = [
            GraphEdge(
                remap[edge.source_id], remap[edge.target_id], edge.relation,
                edge.properties, edge.source, edge.provenance,
            )
            for edge in edges
        ]
        self._store.upsert_nodes(list(canonical.values()))
        self._store.upsert_edges(resolved_edges)
