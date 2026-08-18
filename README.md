# FatBB-agent

An end-to-end system for building a **food-knowledge graph** from a corpus of recipe
Markdown documents: raw recipe pages are turned into structured JSON labels by LLMs,
the labeled data is used to fine-tune a local model, and the accepted labels feed a
multi-backend retrieval foundation (keyword / vector / graph) behind an interactive
terminal client.

The repository is organized into four cooperating packages under [`src/`](src/):

| Package | Purpose |
|---------|---------|
| [`fatbb`](src/fatbb/) | Interactive terminal client for creating, selecting, and querying isolated knowledge bases |
| [`rag`](src/rag/) | Replaceable retrieval foundation: BM25, vector, and knowledge-graph indexing and retrieval |
| [`labeling`](src/labeling/) | LLM labeling pipeline that turns Markdown recipes into structured knowledge-graph JSON |
| [`labeling_sft`](src/labeling_sft/) | QLoRA supervised fine-tuning pipeline that trains a deployable GGUF model on the labeled data |

> **Status — program entry point under construction.** Each pipeline currently runs
> through its own standalone `python -m …` entry script (see [Running the pipelines](#running-the-pipelines)),
> and the `fatbb` terminal client covers knowledge-base retrieval only. A single
> **unified end-to-end program entry point** that ties labeling → fine-tuning → graph
> construction → retrieval together does not exist yet and is still under construction.

---

## System overview

```text
 data/markdown/*.md  (raw recipe corpus)
        │
        ▼
 ┌─────────────────────────────────────────────┐
 │ labeling  (LLM labeling pipeline)           │
 │  bootstrap mode   → OpenAI-compatible cloud │
 │  production mode  → local Ollama model      │
 │  distributed mode → Redis Streams + workers │
 │  SimHash near-duplicate dedup + checkpoint  │
 └──────────────────┬──────────────────────────┘
                    │  accepted labels (raw_text + output JSON)
                    ▼
        dedup_store_bootstrap.sqlite            (authoritative labeled dataset)
        │                        │
        ▼                        ▼
 ┌──────────────────────┐   ┌──────────────────────────────┐
 │ labeling_sft         │   │ rag                          │
 │ QLoRA fine-tune on   │   │ graph build (run_default):   │
 │ accepted labels      │   │ SqliteDedupLoader            │
 │ → GGUF for local use │   │  → GraphIndexer → Neo4j      │
 └──────────────────────┘   └───────┬──────────────────────┘
                                    │
                                    ▼
        rag retrieval: BM25 (pg_search) · Vector (pgvector) · Graph (Neo4j)
        exposed through the fatbb interactive terminal client
```

---

## 1. Interactive terminal client — `fatbb`

`src/fatbb/` implements an English-language terminal UI (built on `prompt-toolkit`)
for managing isolated knowledge bases. It is the only package installed as a console
script (`fatbb = "fatbb.cli.app:main"` in `pyproject.toml`).

### Features

- **Command palette.** Type `/` to open the palette, select **Knowledge Base**, then
  **Select existing** or **Create new**.
- **Knowledge-base creation flow.** A guided, configuration-driven flow (pages, menus,
  routes, and actions are declared in [`src/fatbb/config/cli.toml`](src/fatbb/config/cli.toml)) asks for:
  - retrieval type — **BM25** or **Vector**;
  - database type — **PostgreSQL**;
  - database connection URL;
  - embedding provider / model / URL (Vector only) — **Ollama** + **nomic-embed-text**;
  - source type — **File path** (imports `.md`, `.markdown`, and `.txt` files);
  - knowledge-base name and local source path.
- **Indexing on creation.** The database connection is verified, the PostgreSQL
  migrations are applied, and the local files are imported and indexed. Heavy work runs
  on a background thread so the UI stays responsive, with live progress in the body.
- **Chat retrieval.** Ordinary input is sent as a BM25 (or vector) retrieval query
  scoped to the active knowledge base; results are rendered as scored, citeable
  evidence lines.
- **Isolation.** Every imported document is tagged with its `knowledge_base_id`, so
  retrieval never mixes content across knowledge bases.
- **Local persistence.** Knowledge-base configuration (including each database URL) is
  stored in `~/.fatbb/knowledge_bases.json`, created with owner-only permissions on
  POSIX; operational logs go to `~/.fatbb/fatbb.log`.

### Architecture

The CLI is deliberately split so terminal concerns never leak into application logic:

- **UI events / pure transitions / application use cases / adapters.** [`cli/events.py`](src/fatbb/cli/events.py),
  [`cli/update.py`](src/fatbb/cli/update.py), [`cli/state.py`](src/fatbb/cli/state.py) form a terminal-independent state
  machine; [`cli/controller.py`](src/fatbb/cli/controller.py) bridges terminal events to application use cases;
  [`cli/actions.py`](src/fatbb/cli/actions.py) and [`cli/item_sources.py`](src/fatbb/cli/item_sources.py) are the concrete handlers
  referenced by `cli.toml`.
- **Application service.** [`application/knowledge_base_service.py`](src/fatbb/application/knowledge_base_service.py)
  coordinates persistence, ingestion, indexing, and retrieval use cases.
- **Capability registry.** [`application/registry.py`](src/fatbb/application/registry.py) resolves small stable type
  keys (`bm25`, `vector`, `file_path`) — stored in each knowledge base's local config —
  to concrete adapter factories declared in [`src/fatbb/config/kb.toml`](src/fatbb/config/kb.toml). Adding a new
  capability is a matter of implementing a port and adding a factory line; the CLI does
  not change.
- **Domain ports.** [`domain/ports.py`](src/fatbb/domain/ports.py) defines `KnowledgeBaseAdapter`,
  `SourceImporter`, `KnowledgeBaseRepository`, and `EmbeddingClientFactory` protocols.
- **Infrastructure adapters.** [`infrastructure/kb/bm25.py`](src/fatbb/infrastructure/kb/bm25.py) (PostgreSQL
  `pg_search`), [`infrastructure/kb/vector.py`](src/fatbb/infrastructure/kb/vector.py) (PostgreSQL `pgvector` + Ollama),
  [`infrastructure/importer/local_files.py`](src/fatbb/infrastructure/importer/local_files.py), and
  [`infrastructure/local/local.py`](src/fatbb/infrastructure/local/local.py) (the local catalog).

---

## 2. RAG foundation — `src/rag/`

A replaceable, traceable retrieval foundation. Input is indexed knowledge; output is
scored, **citeable** evidence. The RAG layer never calls an LLM itself (except through
the injected embedding client), and it is structured around narrow interfaces so no
application code depends on a particular database or embedding provider.

### Core models

| File | Types |
|------|-------|
| [`rag/models/common.py`](src/rag/models/common.py) | `Metadata`, `SourceRef` (document id / URI / title / locator) |
| [`rag/models/document.py`](src/rag/models/document.py) | `Document`, `TextChunk`, `ScoredTextChunk` |
| [`rag/models/query.py`](src/rag/models/query.py) | `RetrievalQuery` with `mode ∈ {keyword, vector, graph, hybrid}`, `top_k`, JSONB filters, `entity_ids`, `relation_types`, `max_hops` |
| [`rag/models/evidence.py`](src/rag/models/evidence.py) | `Evidence` — uniform, citeable retrieval output (`text_chunk`, `graph_node`, `graph_edge`, `graph_path`) |
| [`rag/models/graph.py`](src/rag/models/graph.py) | `GraphNode`, `GraphEdge`, `GraphPath`, `ScoredGraphNode`, `slug()`, property/provenance fusion helpers |

### Chunking

[`rag/chunkers/markdown_chunker.py`](src/rag/chunkers/markdown_chunker.py) — heading-aware Markdown chunking:

- splits at ATX headings (ignoring headings inside fenced code blocks), keeps headings
  in the chunk content so lexical retrieval can match them, and records the heading
  hierarchy in each chunk's `heading_path` metadata;
- merges short sections up to a minimum size and splits oversized sections at paragraph
  or whitespace boundaries;
- produces **stable chunk IDs** (`document_id:index:content-hash`) so re-indexing is
  deterministic.

### Lexical retrieval — BM25 (PostgreSQL `pg_search`)

- [`rag/indexers/bm25_indexer.py`](src/rag/indexers/bm25_indexer.py) — `BM25Indexer` chunks documents and atomically
  replaces each document's chunk rows (`replace_document_chunks`) in one connection.
- [`rag/stores/postgres/postgres_bm25_search_store.py`](src/rag/stores/postgres/postgres_bm25_search_store.py) — `PostgresBM25SearchStore`
  persists chunks and executes database-side BM25 queries via the `pg_search` extension
  (e.g. ParadeDB). It creates per-DSN connection pools, validates identifiers, and
  applies the bundled migrations. On Windows it rewrites `localhost` to `127.0.0.1` to
  avoid the IPv6→IPv4 fallback penalty.
- [`rag/retrievers/bm25_retriever.py`](src/rag/retrievers/bm25_retriever.py) — `BM25Retriever` maps backend matches into
  `Evidence` with a `retriever="bm25"` tag.

### Vector retrieval — pgvector

- [`rag/indexers/vector_indexer.py`](src/rag/indexers/vector_indexer.py) — `VectorIndexer` chunks documents; the store
  generates embeddings.
- [`rag/stores/postgres/postgres_vector_search_store.py`](src/rag/stores/postgres/postgres_vector_search_store.py) — `PostgresVectorSearchStore`
  extends the BM25 store with an `embedding` column, cosine-distance search (`<=>`), and
  batched async embedding generation so HTTP round-trips are independent of chunk count.
- [`rag/retrievers/vector_retriever.py`](src/rag/retrievers/vector_retriever.py) — `VectorRetriever` converts vector matches
  into `Evidence`.
- [`rag/client/ollama_embedding_client.py`](src/rag/client/ollama_embedding_client.py) — `OllamaEmbeddingClient` talks to
  Ollama's `/api/embed`, with sub-batching (128 texts/request), bounded concurrency
  (4 parallel requests), retries, and sync/async batch APIs.

### Knowledge-graph retrieval — Neo4j

- [`rag/indexers/graph_indexer.py`](src/rag/indexers/graph_indexer.py) — `GraphIndexer` is a **config-driven** graph builder:
  a JSON schema ([`configs/recipe_graph.json`](configs/recipe_graph.json)) maps an extraction payload's fields to
  node labels (`Dish`, `Ingredient`, `Cuisine`), edge relations (`BELONGS_TO`, `CONTAINS`,
  `VARIANT_OF`, `PAIRS_WITH`, `COMPLEMENTS`, `SUBSTITUTES`, `MAKES`), and property fields.
  It resolves candidates against the store, **fuses duplicates** (first-wins properties,
  alias union, per-property provenance preserved), and upserts idempotently (`MERGE` on
  canonical id). It supports checkpoint files for crash-safe, resumable builds.
- [`rag/stores/neo4j/neo4j_graph_store.py`](src/rag/stores/neo4j/neo4j_graph_store.py) — `Neo4jGraphStore` persists and queries
  the graph. Node ids are `Label:slug`-namespaced so a Dish and an Ingredient can share a
  name without colliding. Entity resolution is lexical first (id, then name/alias,
  case-insensitive, constrained to the same label), with an optional **embedding-based
  fallback** through an HNSW vector index when lexical matching misses (configurable
  `embed_labels`, `embed_threshold`, `embed_dimensions`, `embed_top_k`). Both
  `ensure_constraints()` and `ensure_vector_index()` (Neo4j ≥ 5.11) are idempotent setup
  steps.
- [`rag/loaders/sqlite_dedup.py`](src/rag/loaders/sqlite_dedup.py) — `SqliteDedupLoader` turns accepted rows of the
  labeling dedup store into `Document` values (`raw_text` → content,
  `output` → `metadata["extraction"]`).
- [`rag/run_default.py`](src/rag/run_default.py) — the **knowledge-graph build pipeline**: reads a Neo4j connection
  from `.env.graphrag` (`NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`, `NEO4J_DATABASE`),
  optionally builds an Ollama embedder, loads accepted labels from the dedup SQLite, maps
  them through the schema, and upserts the graph with checkpoint/resume support.

### Database migrations

`migrations/postgres/` contains the SQL templates applied on knowledge-base creation
(validated table identifier substituted at runtime):

- `0001_create_rag_text_chunks.sql` — chunk table, metadata GIN index, `pg_search` BM25 index;
- `0002_add_rag_text_chunk_embeddings.sql` — adds the `vector({{embedding_dimension}})` column;
- `0003_add_rag_text_chunk_embedding_hnsw_index.sql` — resizes the column and creates the
  HNSW vector index.

### Not yet implemented in `rag`

- **Graph retrieval (read side)** — `GraphStore.match_entities()` and `query_paths()`, and a
  `GraphRetriever`, are deferred (the store raises `NotImplementedError` for these).
- **Hybrid retrieval**, reranking, and **context construction** for LLM prompt assembly.

---

## 3. Labeling pipeline — `src/labeling/`

Turns the raw Markdown recipe corpus into structured **food-knowledge-graph JSON labels**
using an LLM. It is resume-safe and deduplicated, and runs in three modes that share the
same ports and persistence.

### Pipeline stages

1. **Discovery & sampling.** [`utils/uri_resolver.py`](src/labeling/utils/uri_resolver.py) globs `**/*.md` and derives a
   stable BLAKE2b `source_id` per file; [`sampling/sampler.py`](src/labeling/sampling/sampler.py) resolves raw text and
   applies **two dedup layers** — a persistent SimHash store and an in-memory
   near-duplicate cluster — yielding batches of `(source_id, hash, raw_text)`.
2. **Checkpointing.** [`checkpoint/file_store.py`](src/labeling/checkpoint/file_store.py) tracks every item through
   `PENDING → IN_FLIGHT → COMPLETED / REJECTED` in a JSON file (atomic writes, Windows-safe).
   On startup, crashed `IN_FLIGHT` items are reset so a run can resume.
3. **LLM labeling.** [`clients/ollama_client.py`](src/labeling/clients/ollama_client.py) (`OllamaLabelingClient`, `/api/chat`)
   and [`clients/openai_client.py`](src/labeling/clients/openai_client.py) (`OpenAILabelingClient`, any OpenAI-compatible
   endpoint) produce an `ExtractionResult` from prompt templates in
   [`src/labeling/prompts/`](src/labeling/prompts/) (`system.txt` + `repair.txt`).
4. **Validation & repair.** [`utils/validator.py`](src/labeling/utils/validator.py) parses model JSON into typed
   dataclasses with two modes: `finetune` (**strict** — used for bootstrap data that will
   train a model) and `production` (**lenient** — enum fallbacks, fuzzy ingredient-ref
   matching). Failed parses can be repaired with a second LLM call.
5. **Deduplication.** [`dedup/simhash_store.py`](src/labeling/dedup/simhash_store.py) computes a recipe-card SimHash
   fingerprint (ingredient + instruction sections, normalized, 64-bit) and stores it in
   SQLite with a **Manku 4×16-bit block index** for fast near-duplicate lookups
   (`hamming distance ≤ threshold`, threshold default 3). Accepted outputs are persisted
   with their raw text, model, and JSON — **this SQLite store is the authoritative
   labeled dataset** consumed by both the SFT pipeline and the RAG graph build.

### Three execution modes

| Mode | Entry point | LLM backend | Notes |
|------|-------------|-------------|-------|
| **Bootstrap** | `python -m labeling.bootstrap.run_default` | OpenAI-compatible cloud (default DeepSeek) | strict `finetune` validation, repair always on; produces the first labeled dataset |
| **Production** | `python -m labeling.pipeline.run_default` | local Ollama fine-tuned model | lenient `production` validation, optional repair; single process |
| **Distributed** | `python -m labeling.pipeline.distribute.run_default --role orchestrator` / `--role worker` | local Ollama | Redis Streams task + result channels, stateless workers |
| **Single-doc debug** | `python -m labeling.pipeline.process_one_default --file <path>` | Ollama | label one document with repair loop, print parsed JSON |

**Distributed mode** splits the production pipeline across a producer/drainer
(`DistributedProductionOrchestrator`, the only writer of dedup/checkpoint state) and
stateless workers (`Worker`, which only pull tasks, call the model, and publish results).
Transport is [`queue/redis_streams.py`](src/labeling/queue/redis_streams.py) (`RedisStreamsWorkQueue`): two streams
(`labeling:tasks`, `labeling:results`) with consumer groups, result published and task
acked in a single transaction, ack-on-read result consumption, and `XAUTOCLAIM`-based
stale-reclaim for crashed workers. Configuration precedence is CLI args > real env >
`.env.orchestrator` / `.env.worker` > code defaults.

### Not yet implemented in `labeling`

- **Confidence scoring** and the human-review CLI described in the pipeline plan
  (`LabelResult.confidence` / `needs_review` are declared but never populated).
- Additional checkpoint/dedup store variants (Memory / SQLite / Redis) — only the file
  checkpoint store and the SimHash SQLite store are implemented.

---

## 4. SFT pipeline — `src/labeling_sft/`

Fine-tunes `Qwen/Qwen2.5-3B-Instruct` (or any configurable base model) with **QLoRA**
(4-bit NF4 quantization + LoRA adapters) on the accepted labels, then exports a
deployable **GGUF** model for local Ollama inference. The pipeline is interface-driven
(dataset build → train → export) behind `SFTOrchestrator`.

### Pipeline stages

1. **Dataset build.** [`dataset_builders/build_from_db/sqlite_local.py`](src/labeling_sft/dataset_builders/build_from_db/sqlite_local.py)
   (`SqliteLocalBuilder`) reads accepted `simhashes` rows (`raw_text` + `output`) from the
   labeling SQLite, shuffles with a fixed seed, splits train/val (default 15%), and writes
   `Alpaca/train.jsonl` + `Alpaca/val.jsonl` in Alpaca format. `BuildFromFileDatasetBuilder`
   builds the same split from a directory of `*.jsonl` files.
2. **Training.** [`trainers/qlora.py`](src/labeling_sft/trainers/qlora.py) (`QLoRATrainer`) loads the JSONL through
   [`dataset_loaders/local_jsonl.py`](src/labeling_sft/dataset_loaders/local_jsonl.py), wraps records in the Qwen chat
   template with the configurable system prompt, applies LoRA to the 4-bit base model, and
   trains with `transformers.Trainer` (completion-only loss masking, GPU-memory logging and
   a memory watchdog callback, tokenized-dataset caching). It auto-resumes from the latest
   `checkpoint-*` and short-circuits when a verified `training_result.json` already exists.
3. **Evaluation.** [`evaluators/ollama_evaluator.py`](src/labeling_sft/evaluators/ollama_evaluator.py) (`OllamaEvaluator`) calls a
   local Ollama server (no in-process model) and scores generations with CPU-only metrics:
   JSON/schema validity, exact match, non-recipe F1, scalar-field accuracy, tags /
   ingredients / step / relations F1. It writes checkpoints so evaluation can resume.
4. **Export.** [`exporters/gguf.py`](src/labeling_sft/exporters/gguf.py) (`GGUFExporter`) merges the LoRA adapter into the
   base model (`merge_and_unload()`), converts it with llama.cpp's `convert_hf_to_gguf.py`
   (auto-cloning llama.cpp on first use), and publishes a GGUF artifact.

### Configuration

Training is configured by a JSON file mapping to [`configs/qlora.py`](src/labeling_sft/configs/qlora.py)
(`QLoRAConfig`): base model, LoRA rank/alpha, 4-bit settings, batch size, learning rate,
`max_seq_length`, scheduler, etc. Example configs are in [`configs/`](configs/):
`run_default_1k.json` (single RTX 4090D 24 GB) and `run_default_1k_h800.json` (H800 80 GB).

### Not yet implemented in `labeling_sft`

- `OllamaEvaluator.compare()` — base-vs-fine-tuned comparison (needs a second Ollama model
  name) raises `NotImplementedError`.
- Non-local data/artifact locations (HF Hub, S3, GCS) raise `NotImplementedError` — the
  default pipeline is local-path only.

---

## Running the pipelines

### Prerequisites

- Python 3.11+
- PostgreSQL with the `pg_search` extension (e.g. ParadeDB) for BM25; `pgvector` for
  vector retrieval
- Neo4j (≥ 5.11 for the HNSW vector index) for graph builds
- Ollama (for embeddings, labeling, and evaluation) and optionally Redis (for distributed
  labeling)

### Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

# Optional: SFT training dependencies
pip install -e '.[finetune]'
```

### Interactive CLI

```bash
fatbb
```

Type `/` to open the command palette and manage knowledge bases (see
[Interactive terminal client](#1-interactive-terminal-client--fatbb)).

### Knowledge-graph build (SQLite labels → Neo4j)

```bash
# Set NEO4J_URI / NEO4J_USERNAME / NEO4J_PASSWORD / NEO4J_DATABASE (or use .env.graphrag)
PYTHONPATH=src python -m rag.run_default
```

### Labeling pipeline

```bash
# Bootstrap: cloud LLM, strict validation, for the first labeled dataset
PYTHONPATH=src python -m labeling.bootstrap.run_default

# Production: local Ollama, single process
PYTHONPATH=src python -m labeling.pipeline.run_default

# Distributed: one orchestrator + any number of stateless workers
PYTHONPATH=src python -m labeling.pipeline.distribute.run_default --role orchestrator
PYTHONPATH=src python -m labeling.pipeline.distribute.run_default --role worker
```

### SFT pipeline

```bash
PYTHONPATH=src python -m labeling_sft.run_default \
  --config configs/run_default_1k.json \
  --database src/labeling/dedup/dedup_store_bootstrap.sqlite \
  --work-dir data/training
```

---

## Tests

```bash
python -m unittest discover -s tests -v
```

Current tests cover the RAG graph indexer/store contracts, labeling orchestrators, the
SimHash dedup store, the Redis Streams queue, the SQLite dedup loader, and the recipe
graph schema. A real PostgreSQL + `pg_search` / Neo4j end-to-end integration test remains
to be added.

## Documentation

- [RAG design and implementation status](docs/rag-design.md)
- [Food knowledge-graph schema design](docs/schema-design.md)
- [Knowledge-graph construction plan](docs/knowledge-graph-plan.md)
- [Knowledge fusion plan](docs/knowledge-fusion-plan.md)
- [Labeling pipeline plan](docs/labeling-pipeline-plan.md)
- [Labeling distributed result-channel design](docs/labeling-distributed-result-channel-design.md)
- [SimHash dedup design](docs/simhash-dedup-design.md)
- [Embedding concurrency notes](docs/embedding-concurrency.md)
- [SFT refactor design](docs/labeling-sft-refactor-design.md)
- [Fine-tuning plan](docs/finetune-plan.md)
- [PostgreSQL migrations](migrations/postgres/)
