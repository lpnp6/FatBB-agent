"""Run the default SQLite → Neo4j knowledge-graph build pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from rag.client.ollama_embedding_client import OllamaEmbeddingClient
from rag.indexers.graph_indexer import GraphIndexer, load_schema
from rag.loaders.sqlite_dedup import SqliteDedupLoader
from rag.stores.neo4j.neo4j_graph_store import Neo4jGraphStore

DEFAULT_DEDUP_DB = Path("src/labeling/dedup/dedup_store_bootstrap.sqlite")
DEFAULT_SCHEMA = Path("configs/recipe_graph.json")
_ENV_FILE = Path(__file__).resolve().parents[2] / ".env.graphrag"


def run_default(
    *,
    uri: str,
    user: str,
    password: str,
    database: str | None,
    dedup_db: Path,
    schema: Path,
    embedding_enabled: bool,
    ollama_model: str,
    ollama_base_url: str,
) -> dict:
    """Assemble the loader/indexer/store and build the graph from SQLite."""
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    embedder = (
        OllamaEmbeddingClient(ollama_model, base_url=ollama_base_url)
        if embedding_enabled else None
    )
    store = Neo4jGraphStore(uri, user, password, embedder, database=database)
    store.ensure_constraints()
    if embedder is not None:
        store.ensure_vector_index()

    documents = SqliteDedupLoader(str(dedup_db)).load()
    GraphIndexer(store, load_schema(schema)).upsert_documents(documents)

    # Validation: count what was written (reuse the store's driver).
    with store._driver.session(database=database) as session:
        nodes = session.run("MATCH (e:Entity) RETURN count(e) AS n").single()["n"]
        edges = session.run("MATCH ()-[r]->() RETURN count(r) AS n").single()["n"]
    return {"documents": len(documents), "nodes": nodes, "edges": edges}


def main() -> None:
    load_dotenv(_ENV_FILE)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dedup-db", type=Path,
        default=Path(os.environ.get("DEDUP_DB") or DEFAULT_DEDUP_DB),
    )
    parser.add_argument(
        "--schema", type=Path,
        default=Path(os.environ.get("SCHEMA_PATH") or DEFAULT_SCHEMA),
    )
    args = parser.parse_args()

    uri = os.environ.get("NEO4J_URI")
    password = os.environ.get("NEO4J_PASSWORD")
    if not uri or not password:
        parser.error("NEO4J_URI and NEO4J_PASSWORD environment variables must be set")

    result = run_default(
        uri=uri,
        user=os.environ.get("NEO4J_USERNAME", "neo4j"),
        password=password,
        database=os.environ.get("NEO4J_DATABASE") or None,
        dedup_db=args.dedup_db,
        schema=args.schema,
        embedding_enabled=os.environ.get("EMBEDDING", "1").lower() not in {"0", "false", "off"},
        ollama_model=os.environ.get("OLLAMA_MODEL", "nomic-embed-text"),
        ollama_base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
