# Knowledge Fusion Plan (recipe graph)

Upgrade the current "lexical blocking + first-wins fusion" toward mainstream
knowledge-fusion practice. Goals: **higher recall** (synonyms/typos merge),
**auditability** (conflicts are not silently dropped), **no over-merging**
(distinct same-named entities are not fused).

## Current state

- **Entity resolution**: `GraphIndexer._resolve_and_upsert` →
  `GraphStore.resolve_entities`. Matching (`_resolve_matches`,
  [neo4j_graph_store.py](../src/rag/stores/neo4j/neo4j_graph_store.py)):
  exact slug → name (case-insensitive) → alias (case-insensitive).
- **Property fusion**: `_merge_nodes` / `_merge_properties`
  ([graph_indexer.py](../src/rag/indexers/graph_indexer.py)).
  List fields union; scalar fields first-non-null (first wins, fill if missing).

## Problems (three failure modes)

| Mode | Symptom | Example |
|---|---|---|
| Over-merge | distinct same-named entities fused | two different recipes both named "Apple Sauce" → `cooking_steps` union corrupts the dish |
| Over-merge | cross-label name collision | "white-rice" as both ingredient and dish; globally unique id allows only one node |
| Under-merge | synonym/spelling variants missed | eggplant/aubergine, caster/castor, chili/chilli, plurals |
| Silent conflict | scalar first-wins drops disagreement | two sources say `cooking_time_min` 30 vs 45; the loser vanishes without trace |

## Plan (three stages)

### Stage 1: keep blocking + add label constraint

Keep slug exact-match as the cheap candidate generator (high precision, zero
cost, interpretable). Add one thing: constrain `resolve_entities` matching by
`label`, separating the ingredient and dish namespaces to kill cross-label
collisions.

- Change: add `AND e.label = $label` to `_resolve_matches` and the Neo4j query
  (or partition by label).
- Cost: one line.
- Benefit: removes the most certain class of over-merge.

### Stage 2: embedding candidate classification (recall)

When lexical matching misses, use vector similarity to recover
synonym/spelling variants.

- **Embedding**: reuse the existing `EmbeddingClient`
  (`OllamaEmbeddingClient`, local nomic-embed-text) — no new dependency, no
  vector database.
- **Embedding input**: the entity `name` alone (single field). Write and
  query must feed the model the same text template (`embed_client.embed(name)`
  on both sides) or the vectors misalign.
- **Retrieval**: Neo4j native HNSW vector index (`db.index.vector.queryNodes`)
  over ``:Entity.embedding``, post-filtered by ``label``.
- **Decision**: high threshold to preserve precision; the borderline band
  (e.g. cosine 0.75–0.9) is re-judged by an LLM.
- **Synonym vs typo split**: embeddings recover **synonyms** (eggplant/
  aubergine, chili/chilli) but are weak on single-char **typos** — BERT-style
  models are insensitive to character noise. Cover typos with a cheap
  character-level fallback (trigram Jaccard or Jaro-Winkler) before/after the
  embedding pass, not with the embedding itself.
- **Scope**: enable for **Ingredient** and **Dish**. Ingredient is a flat
  namespace (synonyms/typos concentrate there, over-merge risk low). Dish
  recall covers variant spellings/translations but carries homonym risk
  (near-name dishes may merge); if that shows up, raise the threshold or use
  a dish-specific threshold.

### Stage 3: conflict-preserving provenance (property fusion)

Replace `_merge_properties`' scalar first-wins with "multi-value + provenance".

- Each property value carries provenance: `source_id` (which document
  contributed it), optionally confidence/time.
- Disagreements are **not** resolved: keep every value, each tagged with its
  source, leaving the conflict to downstream decisions (the Wikidata
  statement + reference pattern).
- Data-model change: property value from bare value → `{value, sources}`, or a
  parallel provenance map.
- Benefit: disagreements are visible and auditable, and the data is ready for
  truth discovery (majority vote / source trust) later.

## Rollout order (by ROI)

1. **Label constraint** (stage 1) — one line; removes a definite over-merge class.
2. **Conflict-preserving provenance** (stage 3) — pure data modeling, highest
   real value, and it paves the way for truth discovery.
3. **Embedding candidate classification** (stage 2) — recall gain;
   Ingredient first, Dish later.

## Non-goals (YAGNI)

- No separate vector database (Neo4j's native HNSW vector index covers ANN).
- No structural / cross-KG entity alignment (no such requirement).
- No truth discovery yet (single source, no conflict data today; add
  provenance first, then truth discovery when the data warrants it).
- No dish-identity upgrade (same-name multi-recipe is a granularity issue, not
  a fusion concern; handled separately).
