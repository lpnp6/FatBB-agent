"""Neo4j implementation of the knowledge-graph storage port."""

from __future__ import annotations

from dataclasses import asdict
import logging
import re
from typing import TYPE_CHECKING

from ...interfaces.stores import GraphStore
from ...models.common import SourceRef
from ...models.graph import GraphEdge, GraphNode, GraphPath, ScoredGraphNode

if TYPE_CHECKING:
    from neo4j import Driver


logger = logging.getLogger(__name__)

_REL_TYPE = re.compile(r"^[A-Z][A-Z0-9_]*$")

# Per-connection drivers shared across store instances (lazy, process-lifetime).
_drivers: dict[tuple[str, str, str], "Driver"] = {}


def _driver(uri: str, user: str, password: str) -> "Driver":
    key = (uri, user, password)
    if key not in _drivers:
        from neo4j import GraphDatabase

        _drivers[key] = GraphDatabase.driver(uri, auth=(user, password))
    return _drivers[key]


def _assert_rel_type(relation: str) -> None:
    """Relationship types are interpolated into Cypher (labels/types are not
    parameterisable); validate against the fixed, config-controlled alphabet."""
    if not _REL_TYPE.match(relation):
        raise ValueError(f"invalid relationship type: {relation!r}")


def _node_params(node: GraphNode) -> dict:
    """Flatten a ``GraphNode`` into Neo4j node properties.

    ``id``/``label``/``name``/``aliases``/``source`` are reserved; domain
    ``properties`` must not collide with them (the schema config guarantees this).
    """
    params = {
        "id": node.id,
        "label": node.label,
        "name": node.name,
        "aliases": list(node.aliases),
    }
    params.update(node.properties)
    if node.source is not None:
        params["source"] = asdict(node.source)
    return params


def _node_from(props: dict) -> GraphNode:
    """Rehydrate a ``GraphNode`` from flat Neo4j properties."""
    label = props.pop("label")
    node_id = props.pop("id")
    name = props.pop("name")
    aliases = tuple(props.pop("aliases", ()))
    source_raw = props.pop("source", None)
    source = SourceRef(**source_raw) if source_raw else None
    return GraphNode(
        id=node_id, label=label, name=name, aliases=aliases,
        properties=props, source=source,
    )


def _edge_params(edge: GraphEdge) -> dict:
    params = dict(edge.properties)
    if edge.source is not None:
        params["source"] = asdict(edge.source)
    return params


def _resolve_matches(
    candidates: list[GraphNode], matched: list[GraphNode],
) -> list[GraphNode | None]:
    """Resolve each candidate to a canonical node: exact id first, then name,
    then alias (case-insensitive) — mirroring the in-memory store's semantics."""
    by_id = {node.id: node for node in matched}
    name_alias: dict[str, GraphNode] = {}
    for node in matched:
        name_alias[node.name.lower()] = node
        for alias in node.aliases:
            name_alias[alias.lower()] = node
    resolved: list[GraphNode | None] = []
    for candidate in candidates:
        node = by_id.get(candidate.id)
        if node is None:
            node = name_alias.get(candidate.name.lower())
            if node is None:
                for alias in candidate.aliases:
                    node = name_alias.get(alias.lower())
                    if node is not None:
                        break
        resolved.append(node)
    return resolved


class Neo4jGraphStore(GraphStore):
    """Persist and query the knowledge graph in Neo4j.

    ``neo4j`` is imported only on first driver use so tests and the build
    pipeline remain importable without the driver installed. Call
    :meth:`ensure_constraints` once at setup to enforce global id uniqueness.
    """

    def __init__(self, uri: str, user: str, password: str):
        self._uri = uri
        self._user = user
        self._password = password

    def ensure_constraints(self) -> None:
        with self._driver.session() as session:
            session.run(
                "CREATE CONSTRAINT entity_id IF NOT EXISTS "
                "FOR (e:Entity) REQUIRE e.id IS UNIQUE"
            )

    # ---- build (GraphIndexer) ------------------------------------------

    def resolve_entities(self, candidates) -> list[GraphNode | None]:
        if not candidates:
            return []
        ids = [candidate.id for candidate in candidates]
        names = {candidate.name.lower() for candidate in candidates}
        for candidate in candidates:
            names.update(alias.lower() for alias in candidate.aliases)
        query = (
            "MATCH (e:Entity) "
            "WHERE e.id IN $ids "
            "   OR toLower(e.name) IN $names "
            "   OR ANY(x IN coalesce(e.aliases, []) WHERE toLower(x) IN $names) "
            "RETURN e"
        )
        with self._driver.session() as session:
            matched = [_node_from(dict(r["e"])) for r in session.run(query, ids=ids, names=list(names))]
        return _resolve_matches(list(candidates), matched)

    def upsert_nodes(self, nodes) -> None:
        if not nodes:
            return
        params = [_node_params(node) for node in nodes]
        with self._driver.session() as session:
            session.run(
                "UNWIND $nodes AS n MERGE (e:Entity {id: n.id}) SET e = n",
                nodes=params,
            )

    def upsert_edges(self, edges) -> None:
        if not edges:
            return
        groups: dict[str, list[dict]] = {}
        for edge in edges:
            groups.setdefault(edge.relation, []).append({
                "source_id": edge.source_id,
                "target_id": edge.target_id,
                "props": _edge_params(edge),
            })
        with self._driver.session() as session:
            for relation, batch in groups.items():
                _assert_rel_type(relation)
                session.run(
                    f"UNWIND $edges AS e "
                    f"MATCH (a:Entity {{id: e.source_id}}), (b:Entity {{id: e.target_id}}) "
                    f"MERGE (a)-[r:{relation}]->(b) SET r += e.props",
                    edges=batch,
                )

    # ---- retrieve (GraphRetriever) --------------------------------------

    def query_nodes(self, entity_ids) -> list[GraphNode]:
        if not entity_ids:
            return []
        with self._driver.session() as session:
            records = session.run(
                "MATCH (e:Entity) WHERE e.id IN $ids RETURN e", ids=list(entity_ids),
            )
            return [_node_from(dict(r["e"])) for r in records]

    def match_entities(self, text, *, top_k) -> list[ScoredGraphNode]:
        # ponytail: needs a fulltext index on name/aliases; part of GraphRetriever.
        raise NotImplementedError("match_entities is deferred to GraphRetriever (P4)")

    def query_paths(self, seeds, *, relation_types, max_hops, top_k) -> list[GraphPath]:
        raise NotImplementedError("query_paths is deferred to GraphRetriever (P4)")
