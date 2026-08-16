# Food Knowledge Graph Construction & Retrieval Plan

> Version: v0.1
> Date: 2026-08-16
> Status: Design (not yet implemented)
> Depends on: [schema-design.md](schema-design.md) (node/relation schema), [rag-design.md](rag-design.md) (retrieval module)

---

## 1. Goal & Current State

We already produce structured food data: the labeling pipeline extracts a per-recipe
JSON payload (Dish / Ingredient / Cuisine nodes + 7 relation types) and persists it in
the dedup store's `simhashes.output` column (SQLite, `ACCEPTED` records only), alongside
`raw_text` and `source_id`. The extraction schema is fully specified in
[schema-design.md](schema-design.md) v1.2.

The missing piece is turning many per-recipe JSON records into **one global,
deduplicated, queryable graph** — i.e. entity resolution (merging the same dish /
ingredient / cuisine mentioned across recipes) plus knowledge fusion (combining
conflicting field values), persisted to a graph database, and exposed through the
existing RAG retrieval port.

## 2. Mainstream Approaches (survey summary)

Three families of KG construction were surveyed (2025–2026):

| Route | Description | Verdict for us |
|---|---|---|
| A. Traditional NER → RE → canonicalization → fusion | Rule/weak-supervision pipeline | Replaced by LLM extraction; not relevant |
| B. LLM extraction + entity resolution + fusion | LLM outputs structured triples; downstream merges them | **This is where we are** |
| C. Agentic / GraphRAG auto-construction (KnoBuilder, AutoSchemaKG) | Agent induces schema and self-corrects | Over-engineered — our schema is already hand-defined |

Key takeaways that shaped this plan:

1. **Entity resolution is the quality lever.** Baseline GraphRAG reaches ~27% noisy-node
   rate; structured prompting + coreference/entity-resolution drops it well below 20%.
   Our extraction already uses a structured prompt; cross-recipe canonicalization is the
   missing half.
2. **Property graph, not RDF**, is the applied/GraphRAG mainstream (Neo4j/Cypher). Our
   schema and Cypher queries ([schema-design.md §9](schema-design.md)) are already
   property-graph shaped.
3. **Fusion needs explicit conflict-resolution rules** (union lists, weight scalars),
   not last-write-wins.

Storage shortlist: **Neo4j Community** (mature, Cypher, GDS) vs **Kuzu** (embedded,
zero-infra, MIT, but original project archived — use the maintained Vela fork). See
[§9 Storage Selection](#9-storage-selection).

## 3. Architecture Decisions

1. **Everything lives in `src/rag/`.** No separate `src/graph/` module. Graph
   construction, storage, and retrieval are all part of the RAG module.
2. **One `GraphStore` interface**, not a write/read split. Both `GraphIndexer` (build)
   and `GraphRetriever` (read) depend on the same single port.
3. **Construction is one `GraphIndexer(Indexer)`** in `indexers/`, mirroring
   `BM25Indexer` / `VectorIndexer`. Mapping + entity-resolution + fusion logic are
   private methods of that indexer (or module-level helpers in the same file), not a
   separate subpackage.
4. **`Neo4jGraphStore` is one class** implementing the full `GraphStore` interface,
   placed under `stores/neo4j/` symmetric to `stores/postgres/`.

> Note: [rag-design.md](rag-design.md) currently states the RAG module "does not …
> build a graph". This plan reverses that; the scope sentence must be updated.

## 4. Module Layout

```text
src/rag/
├── models/
│   ├── graph.py                    # GraphNode / GraphEdge / GraphPath / ScoredGraphNode
│   └── __init__.py                 # re-export graph models
├── interfaces/
│   ├── stores.py                   # + GraphStore (single port)
│   └── __init__.py                 # re-export GraphStore
├── indexers/
│   └── graph_indexer.py            # GraphIndexer(Indexer) — all construction logic
├── retrievers/
│   └── graph_retriever.py          # GraphRetriever(Retriever)
└── stores/
    ├── postgres/                   # (existing text/vector stores)
    └── neo4j/
        ├── __init__.py
        └── neo4j_graph_store.py    # Neo4jGraphStore(GraphStore)
```

## 5. Models (`src/rag/models/graph.py`)

```python
@dataclass(frozen=True)
class GraphNode:
    id: str                              # canonical slug, e.g. "kung-pao-chicken"
    label: str                           # "Dish" | "Ingredient" | "Cuisine"
    name: str                            # canonical English name
    aliases: tuple[str, ...] = ()
    properties: dict[str, object] = field(default_factory=dict)  # food-schema fields
    source: SourceRef | None = None      # provenance (first source_url / source_id)

@dataclass(frozen=True)
class GraphEdge:
    source_id: str
    target_id: str
    relation: str                        # one of the 7 relation types (see §7)
    properties: dict[str, object] = field(default_factory=dict)
    source: SourceRef | None = None

@dataclass(frozen=True)
class GraphPath:
    nodes: tuple[GraphNode, ...]
    edges: tuple[GraphEdge, ...]

@dataclass(frozen=True)
class ScoredGraphNode:
    node: GraphNode
    score: float
```

## 6. Interface (`src/rag/interfaces/stores.py`)

```python
class GraphStore(ABC):
    # ---- build (GraphIndexer) -------------------------------------------
    def resolve_entity(self, name: str, aliases: Sequence[str]) -> str | None:
        """Return the canonical id of an existing entity matching name/aliases
        (L1 slug + L2 alias), or None if this is a new entity."""

    def upsert_node(self, node: GraphNode) -> str:
        """Create or merge a node; returns its canonical id."""

    def upsert_edge(self, edge: GraphEdge) -> None:
        """Create or merge a relationship edge."""

    def commit(self) -> None:
        """Flush buffered writes (used for bulk import)."""

    # ---- retrieve (GraphRetriever) --------------------------------------
    def match_entities(self, text: str, *, top_k: int) -> list[ScoredGraphNode]:
        """Find entities by name/alias/full-text match."""

    def query_nodes(self, entity_ids: Sequence[str]) -> list[GraphNode]:
        """Fetch nodes by canonical id."""

    def query_paths(
        self, seeds: Sequence[str], *,
        relation_types: Sequence[str], max_hops: int, top_k: int,
    ) -> list[GraphPath]:
        """Traverse from seed entities up to max_hops, filtered by relation types."""
```

`GraphIndexer(store: GraphStore)` and `GraphRetriever(store: GraphStore)` both depend on
this one interface. `Neo4jGraphStore` implements all seven methods.

## 7. Construction (`src/rag/indexers/graph_indexer.py`)

`GraphIndexer` implements the existing `Indexer` port:

```python
class GraphIndexer(Indexer):
    def __init__(self, store: GraphStore): ...
    def upsert_documents(self, documents: Sequence[Document], *, on_progress=None) -> None: ...
    def delete_documents(self, document_ids: Sequence[str]) -> None: ...
```

Each `Document` is one recipe record: `content` = raw markdown (provenance),
`metadata["extraction"]` = the parsed `output` JSON dict. The pipeline per document:

1. **Canonicalize** — `name → slug` via a shared `slug()` helper (promoted from
   `scripts/validate_labels.py`). Collect `aliases`.
2. **Entity resolution** — call `store.resolve_entity(name, aliases)` to decide merge
   vs create (L1/L2 below; L3 optional).
3. **Fuse** — apply conflict-resolution rules (§8) before handing values to the store.
4. **Upsert** — `store.upsert_node` / `store.upsert_edge`, then `store.commit()`.

`delete_documents(ids)` removes nodes/edges whose `source.document_id` is in `ids`.

### 7.1 Entity resolution levels

- **L1 — slug primary key**: `slug(name)` is the canonical id. Deterministic, catches
  exact English-name matches.
- **L2 — alias transitive merge**: an `alias → canonical_id` index; if a new node's
  `name` or any `alias` matches an existing node's `name`/`aliases`, merge. This is the
  multilingual-alignment point (`宫保鸡丁` ↔ `Kung Pao Chicken`). Aliases union.
- **L3 — embedding fuzzy match (optional)**: near-miss names ("Chicken Breast" vs
  "Chicken Breast Fillet"). Because the model already emits English canonical names +
  aliases, L1+L2 covers most cases; L3 goes to a **review queue** only, never
  auto-merge.

### 7.2 Relation types (from schema-design.md §6)

`CONTAINS` (Dish→Ingredient), `BELONGS_TO` (Dish→Cuisine), `VARIANT_OF` (Dish↔Dish),
`PAIRS_WITH` (Dish↔Dish), `COMPLEMENTS` / `SUBSTITUTES` / `MAKES` (Ingredient↔Ingredient).

## 8. Fusion Rules

When the same canonical node appears in multiple sources, merge fields as follows:

| Field shape | Rule |
|---|---|
| `name` | keep most frequent (or first-seen) |
| `aliases` | union |
| enum / array (`taste_profile`, `dietary`, `season`) | union (optional per-value source count) |
| scalar (`cooking_time_min`, `calories_per_serving`, `servings`) | keep `{min, max}` range, or confidence-weighted mean |
| `CONTAINS` edge | aggregate per `(dish, ingredient)`; `is_essential` true if any source says true; `amount` retains per-source values |
| relation edges | dedupe by `(from, to, relation)`; union attributes |

## 9. Storage Selection

| | Neo4j Community | Kuzu |
|---|---|---|
| Model / query | property graph / Cypher | property graph / Cypher |
| Deploy | Docker self-host (or Aura cloud) | embedded, in-process, `pip install` |
| Fit | zero migration for our Cypher queries; mature ecosystem; GDS for later analytics | zero infra; MIT; but original project archived (use Vela fork) |
| Verdict | **Recommended** | fallback |

Decision is pending; it only affects the P2 adapter. The `GraphStore` interface keeps the
backend swappable.

## 10. Input Seam (open item)

`Indexer.upsert_documents` accepts text `Document`s, but graph construction needs
structured extraction. Default: structured payload in `Document.metadata["extraction"]`
(dict). Alternative: give `GraphIndexer` a dedicated `upsert_records(Sequence[ExtractionOutput])`
— cleaner typing but no longer literally the `Indexer` interface.

## 11. Phases

| Phase | Work | Deliverable |
|---|---|---|
| P0 | `models/graph.py` + `GraphStore` in `stores.py` + `__init__` re-exports | value objects + port |
| P1 | `indexers/graph_indexer.py` (canonicalize + resolve + fuse), unit-tested with a fake `GraphStore` | pure construction logic |
| P2 | `stores/neo4j/neo4j_graph_store.py` (`MERGE` upsert + resolve + read queries) | storage adapter |
| P3 | application-layer glue: read dedup store `ACCEPTED` → `Document` → `GraphIndexer`; smoke-test with schema-design.md §9's 11 Cypher queries | end-to-end queryable graph |
| P4 | `retrievers/graph_retriever.py` → `Evidence`; later RRF hybrid fusion with BM25/vector | graph retrieval |

## 12. Open Decisions

1. **Storage**: Neo4j Community vs Kuzu (blocks P2).
2. **Input seam**: `metadata["extraction"]` vs dedicated `upsert_records` (default: metadata).

## 13. Sources

- [Wikontic — Wikidata-aligned KG construction](https://huggingface.co/papers/2512.00590)
- [KnoBuilder — agentic KG (NeurIPS 2025)](https://neurips.cc/virtual/2025/loc/mexico-city/129837)
- [AutoSchemaKG](https://huggingface.co/papers/2505.23628)
- [Less is More: Denoising KGs for RAG](https://ar5iv.labs.arxiv.org/html/2510.14271)
- [Kuzu — embedded graph DB (Vela fork)](https://github.com/Vela-Engineering/kuzu)
- [NebulaGraph 2025 roundup](https://www.nebula-graph.com.cn/posts/article-top-2025)
- [Memgraph 3.0](https://www.srmtoday.com/memgraph-launches-3-0-with-new-features-for-generative-ai-applications/)
- [RAGFlow optional entity resolution](https://cloud.tencent.com.cn/developer/article/2515639?policyId=1004)
- [Global GraphRAG & entity resolution](https://blog.csdn.net/2511_93721486/article/details/159555841)
