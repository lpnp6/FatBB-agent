"""Adapter from graph-store retrieval to uniform RAG evidence.

Entity linking (``match_entities``) resolves free text to seed entity ids;
node and subgraph evidence is emitted from ``query_nodes``/``query_paths``.
Scores are the entity-match relevance where known, ``1.0`` for traversals —
ranking across backends is left to a hybrid fuse.

Query the built graph from the command line::

    python -m rag.retrievers.graph_retriever "宫保鸡丁用什么替代鸡肉"

Configuration is read from ``.env.graphrag`` (same keys as ``run_default``).
"""

from __future__ import annotations

from pathlib import Path

from ..interfaces.retriever import Retriever
from ..interfaces.stores import GraphStore
from ..models.evidence import Evidence
from ..models.graph import GraphNode, GraphPath
from ..models.query import RetrievalQuery


def node_text(node: GraphNode) -> str:
    """One-line rendering of a node for LLM context."""
    alias = f" ({', '.join(node.aliases)})" if node.aliases else ""
    props = ", ".join(f"{key}: {value}" for key, value in node.properties.items())
    return f"{node.label}: {node.name}{alias}" + (f" — {props}" if props else "")


def path_text(path: GraphPath) -> str:
    """One-line rendering of a subgraph: node lines, then the edges among them
    with endpoint names (edges are the names, not slugs)."""
    name = {n.id: n.name for n in path.nodes}
    nodes = " | ".join(node_text(n) for n in path.nodes)
    edges = "; ".join(
        f"{name.get(e.source_id, e.source_id)} {e.relation} {name.get(e.target_id, e.target_id)}"
        for e in path.edges
    )
    return f"{nodes} ; {edges}" if edges else nodes


class GraphRetriever(Retriever):
    """Match free text to entities, traverse their neighborhood, emit evidence."""

    def __init__(self, store: GraphStore):
        self._store = store

    def retrieve(self, query: RetrievalQuery) -> list[Evidence]:
        if not query.text.strip() and not query.entity_ids:
            return []
        if query.entity_ids:
            seeds = list(query.entity_ids)
            # Only this path needs a fetch — text matching already returns nodes.
            scored = [(node, 1.0) for node in self._store.query_nodes(seeds)]
        else:
            matched = self._store.match_entities(query.text, top_k=query.top_k)
            seeds = [m.node.id for m in matched]
            scored = [(m.node, m.score) for m in matched]

        seed_scores = {node.id: score for node, score in scored}
        evidence: list[Evidence] = []
        seen: set[str] = set()
        for node, score in scored:
            if node.id in seen:
                continue
            seen.add(node.id)
            evidence.append(Evidence(
                id=f"node:{node.id}",
                kind="graph_node",
                content=node_text(node),
                score=score,
                source=node.source,
                metadata={"retriever": "graph", "label": node.label},
            ))
        for path in self._store.query_paths(
            seeds, relation_types=query.relation_types,
            max_hops=query.max_hops, top_k=query.top_k,
        ):
            pid = "path:" + ",".join(n.id for n in path.nodes)
            if pid in seen:
                continue
            seen.add(pid)
            # A path is a traversal FROM a matched seed: it inherits the seed's
            # relevance so node/path evidence share one comparable score.
            seed_hits = [seed_scores[n.id] for n in path.nodes if n.id in seed_scores]
            evidence.append(Evidence(
                id=pid,
                kind="graph_path",
                content=path_text(path),
                score=max(seed_hits) if seed_hits else 1.0,
                metadata={"retriever": "graph", "edge_count": len(path.edges)},
            ))
        return sorted(evidence, key=lambda e: e.score, reverse=True)


def main() -> None:
    """Query the built graph with a natural-language phrase; print evidence JSON."""
    import argparse
    import json
    import os
    from dataclasses import asdict

    from dotenv import load_dotenv

    from rag.client.ollama_embedding_client import OllamaEmbeddingClient
    from rag.stores.neo4j.neo4j_graph_store import Neo4jGraphStore

    load_dotenv(Path(__file__).resolve().parents[3] / ".env.graphrag")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="+", help="natural-language query")
    parser.add_argument("--top-k", type=int, default=1, help="max evidence items (default: 5)")
    args = parser.parse_args()

    uri = os.environ.get("NEO4J_URI")
    password = os.environ.get("NEO4J_PASSWORD")
    if not uri or not password:
        parser.error("NEO4J_URI and NEO4J_PASSWORD must be set in .env.graphrag")

    embedding_enabled = os.environ.get("EMBEDDING", "1").lower() not in {"0", "false", "off"}
    embedder = (
        OllamaEmbeddingClient(
            os.environ.get("OLLAMA_MODEL", "nomic-embed-text"),
            base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
        )
        if embedding_enabled else None
    )
    store = Neo4jGraphStore(
        uri, os.environ.get("NEO4J_USERNAME", "neo4j"), password, embedder,
        database=os.environ.get("NEO4J_DATABASE") or None,
    )
    store.ensure_fulltext_index()
    if embedder is not None:
        store.ensure_vector_index()

    evidence = GraphRetriever(store).retrieve(
        RetrievalQuery(text=" ".join(args.query), top_k=args.top_k, mode="graph"),
    )
    print(json.dumps([
        {
            "id": e.id,
            "kind": e.kind,
            "score": e.score,
            "content": e.content,
            "source": asdict(e.source) if e.source is not None else None,
            "metadata": e.metadata,
        }
        for e in evidence
    ], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
