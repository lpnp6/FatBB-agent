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
```

The migration creates the `rag_text_chunks` table, metadata indexes, and the BM25 index. The application does not execute DDL at runtime.

## Interactive CLI

After installing the project and applying the RAG migration, start a persistent terminal session:

```bash
fatbb
```

The UI is entirely in English. Type `/` to open the command palette and select **Knowledge Base**. Press Backspace to delete `/` and return to the chat input. The knowledge-base creation flow currently exposes one selectable default at each capability step: **BM25**, **PostgreSQL**, and **File path**. It asks for the PostgreSQL connection URL during creation, stores that URL in the local knowledge-base catalog, indexes supported local `.md`, `.markdown`, and `.txt` files, then sends ordinary chat input to BM25 retrieval scoped to the selected knowledge base. Existing knowledge bases are listed immediately from that local catalog.

Knowledge-base configuration, including its PostgreSQL URL, is stored locally at `~/.fatbb/knowledge_bases.json`. On POSIX systems FatBB creates the containing directory with `700` permissions and the catalog with `600` permissions. The URL may contain credentials, so do not share this file. Indexed chunks are tagged with `knowledge_base_id` metadata, so retrieval never mixes content from separate knowledge bases. The CLI is structured around UI events, pure state transitions, application use cases, and registered knowledge-base/source adapters.

## Capability registration

`src/fatbb/config/kb.toml` maps stable knowledge-base identifiers to adapter factories and is committed with the code. A knowledge base persists only identifiers such as `bm25` and `file_path` in its local configuration. When it is used later, `CapabilityRegistry` resolves those identifiers and builds the configured adapter. Terminal navigation, menus, and creation-flow actions are kept separately in `src/fatbb/config/cli.toml`; the CLI does not import KB modules.

To add a knowledge-base capability, implement the relevant domain port and add its factory to `kb.toml`:

```python
[knowledge_bases.vector]
label = "Vector"
factory = "fatbb.infrastructure.kb.vector:VectorKnowledgeBase"
```

The CLI and knowledge-base workflow do not need to change.

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
