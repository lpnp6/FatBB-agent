# FatBB-agent

This repository contains a replaceable RAG foundation. Its first production-facing implementation is BM25 text retrieval backed by PostgreSQL `pg_search`: source documents are split into chunks, persisted to PostgreSQL, and ranked by the database. Query results are returned as citeable `Evidence` objects. It also provides an interactive English-language `fatbb` terminal client for creating, selecting, and querying isolated knowledge bases.

## Current capabilities

- Heading-aware Markdown chunking with stable chunk IDs.
- `BM25Indexer` for document creation, replacement, and deletion.
- `PostgresTextChunkStore` for PostgreSQL persistence and database-side BM25 search.
- `BM25Retriever` for converting scored matches into `Evidence`.
- Equality filtering through JSONB metadata.

Vector retrieval, graph retrieval, hybrid retrieval, context construction, and an application service entry point are not implemented yet.

## Architecture

```text
Document
  -> MarkdownChunker
  -> BM25Indexer
  -> PostgresTextChunkStore.replace_document_chunks()
  -> PostgreSQL + pg_search BM25 index

RetrievalQuery
  -> BM25Retriever
  -> PostgresTextChunkStore.search_bm25()
  -> Evidence[]
```

`BM25Indexer` and `BM25Retriever` depend only on the `BM25SearchStore` interface, not on PostgreSQL. A different search backend can be used by implementing that interface and injecting it at application composition time.

## Prerequisites

You need a PostgreSQL instance that provides the `pg_search` extension (for example, ParadeDB) and Python 3.11 or later.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

export DATABASE_URL='postgresql://user:password@localhost:5432/fatbb'
psql "$DATABASE_URL" -f migrations/postgres/0001_create_rag_text_chunks.sql
psql "$DATABASE_URL" -f migrations/postgres/0002_create_knowledge_bases.sql
```

The migration creates the `rag_text_chunks` table, metadata indexes, and the BM25 index. The application does not execute DDL at runtime.

## Interactive CLI

After installing the project and applying both migrations, start a persistent terminal session:

```bash
fatbb
```

The UI is entirely in English. Type `/` to open the command palette and select **Knowledge Base**. Press Backspace to delete `/` and return to the chat input. The knowledge-base creation flow currently exposes one selectable default at each capability step: **BM25**, **PostgreSQL**, and **File path**. It indexes supported local `.md`, `.markdown`, and `.txt` files, then sends ordinary chat input to BM25 retrieval scoped to the selected knowledge base.

Knowledge-base configuration is stored in `knowledge_bases`; indexed chunks are tagged with `knowledge_base_id` metadata, so retrieval never mixes content from separate knowledge bases. The CLI is structured around UI events, pure state transitions, application use cases, and registered backend/source adapters. To add a future retriever or source type, implement and register a `RetrievalBackend` or `SourceImporter` without changing terminal input handling.

## Indexing and retrieval

Run an example from the repository root with `src` on the Python import path:

```bash
PYTHONPATH=src python example.py
```

The essential application wiring is:

```python
from rag.chunkers import MarkdownChunker
from rag.indexers import BM25Indexer
from rag.models import Document, RetrievalQuery, SourceRef
from rag.retrievers import BM25Retriever
from rag.stores import PostgresTextChunkStore

store = PostgresTextChunkStore("postgresql://user:password@localhost:5432/fatbb")
indexer = BM25Indexer(MarkdownChunker(), store)
retriever = BM25Retriever(store)

indexer.upsert_documents([
    Document(
        id="postgres-guide",
        content="# PostgreSQL\n\nPostgreSQL supports full-text search.",
        source=SourceRef(uri="https://example.com/postgres", title="PostgreSQL guide"),
        metadata={"tenant_id": "acme", "language": "en"},
    )
])

evidence = retriever.retrieve(
    RetrievalQuery(
        text="PostgreSQL full-text search",
        mode="keyword",
        top_k=5,
        filters={"tenant_id": "acme"},
    )
)

for item in evidence:
    print(item.score, item.content, item.source)
```

Calling `upsert_documents()` again with the same `Document.id` atomically replaces all chunks for that document. To delete a document, call `indexer.delete_documents(["postgres-guide"])`.

## Tests

```bash
python -m unittest discover -s tests -v
```

The current tests cover domain models, Markdown chunking, and indexer/retriever/store contracts. A real PostgreSQL + `pg_search` end-to-end integration test remains to be added.

## Documentation

- [RAG design and implementation status](docs/rag-design.md)
- [PostgreSQL BM25 migration](migrations/postgres/0001_create_rag_text_chunks.sql)
