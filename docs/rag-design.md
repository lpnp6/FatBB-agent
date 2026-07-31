# RAG Module Design

## Scope and implementation status

The RAG module provides replaceable, traceable knowledge retrieval for higher-level services. Its input is indexed knowledge and its output is scored, citeable evidence. It does not call an LLM, manage conversation memory, collect source material, or build a graph.

The implemented path is PostgreSQL `pg_search`-backed BM25 text retrieval:

```text
Document
  -> MarkdownChunker
  -> BM25Indexer
  -> BM25SearchStore.replace_document_chunks()
  -> PostgresTextChunkStore
  -> PostgreSQL + pg_search BM25 index

RetrievalQuery
  -> BM25Retriever
  -> BM25SearchStore.search_bm25()
  -> PostgresTextChunkStore
  -> Evidence[]
```

The following capabilities are design targets only: vector retrieval, graph retrieval, hybrid retrieval, reranking, and context construction.

## Design principles

1. Return evidence rather than raw documents. A text chunk, graph node, edge, or path can support an answer.
2. Isolate infrastructure behind narrow interfaces. Application code must not depend on a particular database or embedding provider.
3. Preserve traceability. Every evidence item must point back to its source document, URI, or graph source.
4. Keep writes separate from reads. `Retriever` is read-only; chunking, indexing, and deletion are performed by `Indexer` and storage implementations.
5. Keep retrieval results separate from prompt construction. Retrieval retains structured data; a future context builder will enforce token budgets and deduplication.

## Current module layout

```text
src/rag/
├── models/
│   ├── common.py                 # Metadata and SourceRef
│   ├── document.py               # Document and TextChunk
│   ├── evidence.py               # Evidence
│   └── query.py                  # RetrievalQuery
├── interfaces/
│   ├── chunker.py                # Document-to-TextChunk port
│   ├── indexer.py                # Index write and delete port
│   ├── retriever.py              # Read-only retrieval port
│   └── stores.py                 # Text chunk and BM25 storage ports
├── chunkers/markdown_chunker.py
├── indexers/bm25_indexer.py
├── retrievers/bm25_retriever.py
└── stores/postgres_text_chunk_store.py
```

`BM25Indexer` and `BM25Retriever` both depend on `BM25SearchStore`. They do not import a database SDK. `PostgresTextChunkStore` is the PostgreSQL-specific adapter injected by the application composition layer.

## Domain model

```python
@dataclass(frozen=True)
class SourceRef:
    document_id: str | None = None
    uri: str | None = None
    title: str | None = None
    locator: str | None = None


@dataclass(frozen=True)
class Document:
    id: str
    content: str
    source: SourceRef
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class TextChunk:
    id: str
    document_id: str
    content: str
    index: int
    source: SourceRef
    start_offset: int | None = None
    end_offset: int | None = None
    metadata: dict[str, object] = field(default_factory=dict)
```

`Document` is an original source and is normally not retrieved directly. `TextChunk` is the smallest unit indexed and retrieved by BM25. A chunk preserves its document source and metadata. Its ID must be stable for identical document content and chunking results.

```python
@dataclass(frozen=True)
class RetrievalQuery:
    text: str
    top_k: int = 5
    mode: Literal["keyword", "vector", "graph", "hybrid"] = "hybrid"
    filters: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class Evidence:
    id: str
    kind: Literal["text_chunk", "graph_node", "graph_edge", "graph_path"]
    content: str
    score: float
    source: SourceRef | None = None
    metadata: dict[str, object] = field(default_factory=dict)
    chunk: TextChunk | None = None
```

For `text_chunk` evidence, `chunk` is required. `BM25Retriever` adds `retriever="bm25"` and the backend score as `raw_score` to the evidence metadata.

## Interfaces

```python
class Indexer(ABC):
    @abstractmethod
    def upsert_documents(self, documents: Sequence[Document]) -> None: ...

    @abstractmethod
    def delete_documents(self, document_ids: Sequence[str]) -> None: ...


class Retriever(ABC):
    @abstractmethod
    def retrieve(self, query: RetrievalQuery) -> list[Evidence]: ...
```

The text storage boundary has lifecycle operations, while the BM25 specialization adds native search:

```python
class TextChunkStore(ABC):
    def list_chunks(self, *, filters: Mapping[str, object]) -> list[TextChunk]: ...
    def upsert_chunks(self, chunks: Sequence[TextChunk]) -> None: ...
    def delete_chunks(self, chunk_ids: Sequence[str]) -> None: ...
    def replace_document_chunks(
        self, document_id: str, chunks: Sequence[TextChunk]
    ) -> None: ...
    def delete_by_document_ids(self, document_ids: Sequence[str]) -> None: ...


class BM25SearchStore(TextChunkStore):
    def search_bm25(
        self,
        query_text: str,
        *,
        top_k: int,
        filters: Mapping[str, object],
    ) -> list[ScoredTextChunk]: ...
```

`BM25Indexer` chunks each document and calls `replace_document_chunks()`. It validates that each chunk belongs to the document being written. `BM25Retriever` short-circuits blank queries, delegates ranking to `search_bm25()`, and wraps each scored chunk as `Evidence`.

## PostgreSQL BM25 adapter

`PostgresTextChunkStore` uses psycopg 3 and implements `BM25SearchStore`. It performs the following operations:

- persists chunk fields, source references, and metadata in `rag_text_chunks`;
- atomically replaces every chunk for one document, removing obsolete chunks;
- deletes chunks by ID or document ID;
- executes `pg_search` BM25 queries in the database;
- applies JSONB metadata filters before ranking, sorts by backend score, and limits results in the database.

Database objects are created from versioned migrations when a knowledge base is created.

The migration templates receive the configured store table name and create the
`pg_search` extension, table, indexes for document IDs and metadata, and the
BM25 index. The runtime connection requires a PostgreSQL environment that
supports `pg_search`, such as ParadeDB.

## Operational behavior

Metadata filters are flat, serializable equality constraints such as `tenant_id`, `category`, `language`, or an authorization scope. They are applied in the storage query rather than after retrieval.

Reindexing an existing document is idempotent with respect to the emitted chunks: chunks no longer produced by the chunker are removed in the same storage transaction. Deleting a source document removes all of its chunks.

The store lazily imports `psycopg`, allowing pure model and contract tests to run without importing the database driver during module loading.

## Testing and gaps

Current unit tests cover:

- Markdown chunk boundaries, stable IDs, and source/metadata inheritance;
- `BM25Indexer` chunk replacement and document deletion delegation;
- `BM25Retriever` empty queries, metadata forwarding, result conversion, and ordering;
- PostgreSQL row conversion and domain-model validation.

The test suite currently uses fake storage adapters for indexing and retrieval. A real PostgreSQL + `pg_search` integration test that indexes a document and retrieves it remains to be added.

## Planned evolution

Future adapters should follow the same separation of responsibilities:

- a vector retriever can combine an `EmbeddingProvider` with a `VectorStore`;
- a graph retriever can query a dedicated `GraphStore` and return nodes, edges, or paths;
- a hybrid retriever can fuse independently retrieved evidence, for example with Reciprocal Rank Fusion;
- a context builder can convert evidence to prompt text under token and source-diversity constraints.

These additions should not change the `Document`, `Evidence`, `Indexer`, or `Retriever` boundaries used by the BM25 implementation.
