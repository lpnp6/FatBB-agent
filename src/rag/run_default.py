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

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DEDUP_DB = Path("src/labeling/dedup/dedup_store_bootstrap.sqlite")
DEFAULT_SCHEMA = Path("configs/recipe_graph.json")
_ENV_FILE = _REPO_ROOT / ".env.graphrag"


def _repo_path(value: str | Path) -> Path:
    """Interpret an env-supplied path relative to the repo root.

    A leading ``/`` (common in env files on Windows, e.g. ``/data/build``)
    means "repo-relative", not "drive-root": ``Path('/data/build')`` would
    otherwise resolve to ``D:\\data\\build`` at the drive root.
    """
    path = Path(value)
    posix = path.as_posix()
    # Rooted-but-driveless (e.g. \data\build) is NOT "absolute" per pathlib,
    # yet joining it discards the prefix — so strip the root first.
    if not path.drive and posix.startswith("/") and not posix.startswith("//"):
        path = Path(posix.lstrip("/"))
    return path if path.is_absolute() else _REPO_ROOT / path


def run_default(
    *,
    uri: str,
    user: str,
    password: str,
    database: str | None,
    dedup_db: Path,
    schema: Path,
    checkpoint: Path | None,
    log_dir: Path | None,
    embedding_enabled: bool,
    ollama_model: str,
    ollama_base_url: str,
) -> dict:
    """Assemble the loader/indexer/store and build the graph from SQLite."""
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_dir / "graph_build.log", encoding="utf-8"))
    logging.basicConfig(
        level="INFO", format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
    )

    embedder = (
        OllamaEmbeddingClient(ollama_model, base_url=ollama_base_url)
        if embedding_enabled else None
    )
    store = Neo4jGraphStore(uri, user, password, embedder, database=database)
    store.ensure_constraints()
    store.ensure_fulltext_index()
    if embedder is not None:
        store.ensure_vector_index()

    documents = SqliteDedupLoader(str(dedup_db)).load()
    GraphIndexer(store, load_schema(schema)).upsert_documents(
        documents, checkpoint=checkpoint,
    )

    # Validation: count what was written (reuse the store's driver).
    with store._driver.session(database=database) as session:
        nodes = session.run("MATCH (e:Entity) RETURN count(e) AS n").single()["n"] # type: ignore
        edges = session.run("MATCH ()-[r]->() RETURN count(r) AS n").single()["n"] # type: ignore
    return {"documents": len(documents), "nodes": nodes, "edges": edges}


def main() -> None:
    load_dotenv(_ENV_FILE)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dedup-db", type=Path,
        default=_repo_path(os.environ.get("DEDUP_DB") or DEFAULT_DEDUP_DB),
    )
    parser.add_argument(
        "--schema", type=Path,
        default=_repo_path(os.environ.get("SCHEMA_PATH") or DEFAULT_SCHEMA),
    )
    parser.add_argument(
        "--checkpoint", type=Path,
        default=_repo_path(os.environ.get("CHECKPOINT_PATH", "build_checkpoint.json")),
        help="resume-from file recording completed document ids (default: build_checkpoint.json)",
    )
    parser.add_argument(
        "--log-dir", type=Path,
        default=_repo_path(os.environ["LOG_DIR"]) if os.environ.get("LOG_DIR") else None,
        help="directory to write graph_build.log (default: logs go to stderr only)",
    )
    args = parser.parse_args()

    # Leniency: a checkpoint that names a directory (no extension) means
    # "put build_checkpoint.json inside it", e.g. CHECKPOINT_PATH=/data/build/.
    if not args.checkpoint.suffix:
        args.checkpoint = args.checkpoint / "build_checkpoint.json"

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
        checkpoint=args.checkpoint,
        log_dir=args.log_dir,
        embedding_enabled=os.environ.get("EMBEDDING", "1").lower() not in {"0", "false", "off"},
        ollama_model=os.environ.get("OLLAMA_MODEL", "nomic-embed-text"),
        ollama_base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
