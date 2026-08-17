"""Neo4j implementation of the knowledge-graph storage port."""

from __future__ import annotations

from dataclasses import asdict
import logging
import re
from typing import TYPE_CHECKING

from ...interfaces.client import EmbeddingClient
from ...interfaces.stores import GraphStore
from ...models.common import SourceRef
from ...models.graph import (
    GraphEdge,
    GraphNode,
    GraphPath,
    ScoredGraphNode,
    merge_properties,
    merge_provenance,
)

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

    ``id``/``label``/``name``/``aliases``/``source``/``provenance`` are reserved;
    domain ``properties`` must not collide with them (the schema config
    guarantees this).
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
    if node.provenance:
        params["provenance"] = node.provenance
    return params


def _node_from(props: dict) -> GraphNode:
    """Rehydrate a ``GraphNode`` from flat Neo4j properties."""
    label = props.pop("label")
    node_id = props.pop("id")
    name = props.pop("name")
    aliases = tuple(props.pop("aliases", ()))
    source_raw = props.pop("source", None)
    props.pop("embedding", None)  # reserved; read separately for vector matching
    provenance = props.pop("provenance", None) or {}
    source = SourceRef(**source_raw) if source_raw else None
    return GraphNode(
        id=node_id, label=label, name=name, aliases=aliases,
        properties=props, source=source, provenance=provenance,
    )


def _edge_params(edge: GraphEdge) -> dict:
    params = dict(edge.properties)
    if edge.source is not None:
        params["source"] = asdict(edge.source)
    if edge.provenance:
        params["provenance"] = edge.provenance
    return params


_EDGE_RESERVED = ("source", "provenance")


def _split_edge(props: dict) -> tuple[dict, dict | None, dict]:
    """Split relationship props into ``(domain, source, provenance)``."""
    domain = {k: v for k, v in props.items() if k not in _EDGE_RESERVED}
    return domain, props.get("source"), props.get("provenance") or {}


def _merge_edge(existing_props: dict, edge: GraphEdge) -> dict:
    """Merge ``edge`` into an existing relationship's props, preserving provenance."""
    domain, source, provenance = _split_edge(existing_props)
    params = merge_properties(domain, edge.properties)
    if source is not None:
        params["source"] = source
    elif edge.source is not None:
        params["source"] = asdict(edge.source)
    merged_provenance = merge_provenance(provenance, edge.provenance)
    if merged_provenance:
        params["provenance"] = merged_provenance
    return params


def _resolve_matches(
    candidates: list[GraphNode], matched: list[GraphNode],
) -> list[GraphNode | None]:
    """Resolve each candidate to a canonical node: exact id first, then name,
    then alias (case-insensitive) — constrained to the same ``label`` so a
    Dish and an Ingredient sharing a name never cross-resolve."""
    by_id = {node.id: node for node in matched}
    name_alias: dict[str, dict[str, GraphNode]] = {}
    for node in matched:
        name_alias.setdefault(node.name.lower(), {})[node.label] = node
        for alias in node.aliases:
            name_alias.setdefault(alias.lower(), {})[node.label] = node
    resolved: list[GraphNode | None] = []
    for candidate in candidates:
        node = by_id.get(candidate.id)
        if node is not None and node.label != candidate.label:
            node = None
        if node is None:
            node = name_alias.get(candidate.name.lower(), {}).get(candidate.label)
        if node is None:
            for alias in candidate.aliases:
                node = name_alias.get(alias.lower(), {}).get(candidate.label)
                if node is not None:
                    break
        resolved.append(node)
    return resolved


class Neo4jGraphStore(GraphStore):
    """Persist and query the knowledge graph in Neo4j.

    ``neo4j`` is imported only on first driver use so tests and the build
    pipeline remain importable without the driver installed. Call
    :meth:`ensure_constraints` once at setup to enforce global id uniqueness.

    An optional ``embedder`` enables embedding-based entity resolution: names
    of ``embed_labels`` nodes are embedded at upsert and recovered via the
    HNSW vector index when lexical matching misses. Call
    :meth:`ensure_vector_index` once at setup (Neo4j ≥5.11).
    """

    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        embedder: EmbeddingClient | None = None,
        *,
        embed_threshold: float = 0.85,
        embed_labels: tuple[str, ...] = ("Ingredient", "Dish"),
        embed_dimensions: int = 768,
        embed_top_k: int = 20,
    ):
        self._uri = uri
        self._user = user
        self._password = password
        self._driver = _driver(uri, user, password)
        self._embedder = embedder
        self._embed_threshold = embed_threshold
        self._embed_labels = embed_labels
        if embed_dimensions <= 0:
            raise ValueError("embed_dimensions must be positive")
        self._embed_dimensions = embed_dimensions
        self._embed_top_k = embed_top_k

    def ensure_constraints(self) -> None:
        with self._driver.session() as session:
            session.run(
                "CREATE CONSTRAINT entity_id IF NOT EXISTS "
                "FOR (e:Entity) REQUIRE e.id IS UNIQUE"
            )

    def ensure_vector_index(self) -> None:
        """Create the HNSW vector index over ``:Entity.embedding`` (Neo4j ≥5.11).

        Call once at setup, after ``ensure_constraints``. The index is keyed on
        the shared ``:Entity`` label, so matches are post-filtered by ``label``.
        """
        query = (
            "CREATE VECTOR INDEX entity_embedding IF NOT EXISTS "
            "FOR (e:Entity) ON (e.embedding) "
            "OPTIONS {indexConfig: {`vector.dimensions`: "
            f"{self._embed_dimensions}, "
            "`vector.similarityFunction`: 'cosine'}}"
        )
        with self._driver.session() as session:
            session.run(query)
            session.run("CALL db.awaitIndexes()")

    # ---- build (GraphIndexer) ------------------------------------------

    def resolve_entities(self, candidates) -> list[GraphNode | None]:
        if not candidates:
            return []
        candidates = list(candidates)
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
        resolved = _resolve_matches(candidates, matched)
        self._embedding_fallback(candidates, resolved)
        return resolved

    def upsert_nodes(self, nodes) -> None:
        if not nodes:
            return
        nodes = list(nodes)
        params = [_node_params(node) for node in nodes]
        if self._embedder is not None:
            embed_idx = [i for i, node in enumerate(nodes) if node.label in self._embed_labels]
            if embed_idx:
                vectors = self._embedder.batch_embedding([nodes[i].name for i in embed_idx])
                for i, vector in zip(embed_idx, vectors):
                    params[i]["embedding"] = vector
        with self._driver.session() as session:
            session.run(
                "UNWIND $nodes AS n MERGE (e:Entity {id: n.id}) SET e = n",
                nodes=params,
            )

    def upsert_edges(self, edges) -> None:
        if not edges:
            return
        edges = list(edges)
        groups: dict[str, list[GraphEdge]] = {}
        for edge in edges:
            groups.setdefault(edge.relation, []).append(edge)
        with self._driver.session() as session:
            for relation, batch in groups.items():
                _assert_rel_type(relation)
                keys = [
                    {"source_id": edge.source_id, "target_id": edge.target_id}
                    for edge in batch
                ]
                existing: dict[tuple[str, str], dict] = {}
                for record in session.run(
                    f"UNWIND $keys AS k "
                    f"MATCH (a:Entity {{id: k.source_id}})-[r:{relation}]->(b:Entity {{id: k.target_id}}) "
                    f"RETURN k.source_id AS source_id, k.target_id AS target_id, properties(r) AS props",
                    keys=keys,
                ):
                    existing[(record["source_id"], record["target_id"])] = dict(record["props"])
                merged: dict[tuple[str, str], dict] = {}
                for edge in batch:
                    key = (edge.source_id, edge.target_id)
                    base = merged.get(key, existing.get(key))
                    merged[key] = _merge_edge(base, edge) if base is not None else _edge_params(edge)
                session.run(
                    f"UNWIND $edges AS e "
                    f"MATCH (a:Entity {{id: e.source_id}}), (b:Entity {{id: e.target_id}}) "
                    f"MERGE (a)-[r:{relation}]->(b) SET r = e.props",
                    edges=[
                        {"source_id": source_id, "target_id": target_id, "props": props}
                        for (source_id, target_id), props in merged.items()
                    ],
                )

    def _embedding_fallback(
        self, candidates: list[GraphNode], resolved: list[GraphNode | None],
    ) -> None:
        """Recover synonyms for unresolved candidates via the HNSW vector index.

        Restricted to labels in ``self._embed_labels``; other labels stay
        lexical. See docs/knowledge-fusion-plan.md stage 2 — the borderline
        LLM re-judge and character-level typo fallback are deferred.
        """
        pending = [
            i for i, (candidate, node) in enumerate(zip(candidates, resolved))
            if node is None and candidate.label in self._embed_labels
        ]
        if not pending:
            return
        embedder = self._embedder
        if embedder is None:
            return
        query_vectors = embedder.batch_embedding(
            [candidates[i].name for i in pending],
        )
        for index, vector in zip(pending, query_vectors):
            match = self._vector_match(candidates[index].label, vector)
            if match is not None:
                resolved[index] = match

    def _vector_match(self, label: str, vector: list[float]) -> GraphNode | None:
        """Top same-label node above the threshold via ANN, or ``None``.

        ``db.index.vector.queryNodes`` searches the shared ``:Entity`` index, so
        the result is post-filtered by ``label``. ``embed_top_k`` must be large
        enough that the target label survives the filter.
        """
        query = (
            "CALL db.index.vector.queryNodes('Entity', 'embedding', $top_k, $vector) "
            "YIELD node, score "
            "WHERE node.label = $label AND score >= $threshold "
            "RETURN node ORDER BY score DESC LIMIT 1"
        )
        with self._driver.session() as session:
            record = session.run(
                query,
                top_k=self._embed_top_k,
                vector=vector,
                label=label,
                threshold=self._embed_threshold,
            ).single()
        return _node_from(dict(record["node"])) if record else None

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
