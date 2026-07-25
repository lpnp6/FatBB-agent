# Bootstrap Labeling Pipeline Plan

## Context

FatBB-agent needs 500–1000 seed labels for fine-tuning Qwen2.5-3B-Instruct. We have 5,564 English recipe Markdown files from RecipeTin Eats and Well Plated. The workflow: **sample → dedup → prompt → label via API → validate → review → QLoRA dataset**.

**Critical architectural requirement**: the labeling pipeline must be **model-agnostic**. The same orchestration code should drive both the Claude API bootstrap phase AND the local fine-tuned model in production. This is achieved via an abstract `LabelingClient` interface with swappable backends (see Architecture section).

## Key Data Insights (from corpus analysis)

| Fact | Implication |
|------|-------------|
| 927 files are `comment-page-N` near-duplicates | Same recipe body, different comments — need to deduplicate before sampling |
| ~1,473 files are `wprm_print` printer-friendly variant | **Preferred input** — explicit times, servings, nutrition, no blog fluff, no comments |
| Cuisine / taste / dietary are only in blog prose | wprm_print alone misses these; need **hybrid**: full-page for prose, wprm_print for nutrition |
| Recipe card (`### Ingredients` / `### Instructions`) structure is consistent | Prompt can rely on this marker pattern |
| Nutrition data only in wprm_print | Use wprm_print variant when available |
| Non-recipe pages exist (roundups, gift guides, etc.) | Additional filter: require both `### Ingredients` AND `### Instructions` headers |

## Sampling Strategy

### preprocessor.py: Filter Non-Recipe Pages

A single concrete module. Two-pass classification — fast rules first, heuristics on survivors:

**Pass 1: Rule-based** — filename patterns, file size, structure markers:
  - Exclude by filename pattern: comment-page-N, catch-up, gift-guide, about,
    holiday, menu, roundup, weekly-meal-plan, christmas-selfie, dozer-update,
    new-recipes-were-loving, essential-salad-dressings, spices-for,
    mothers-day-breakfasts, creative-bites, asian-meals-on-the-table
  - Exclude by size: file < 2KB (cookbook teasers, page fragments)
  - Exclude by structure: file missing `### Ingredients` OR `### Instructions`

**Pass 2: Heuristic** — content analysis catches what rules miss:
  - Files with >3 recipe-like headings = roundup
  - Files where intro text > 80% of total content = blog post
  - Ingredients section with >50 bullet points = directory/roundup

Output: `data/preprocessor_report.json`
  - total: 5564
  - recipes: ~2900 (estimated)
  - non_recipe breakdown by reason
```

### DedupStore: Persistent Content-Level De-duplication

Content-level duplicates survive the preprocessor (same content, different filenames). A persistent `DedupStore` module prevents them from ever reaching the API:

| Duplicate scenario | Example | Detection method |
|---|---|---|
| wprm_print vs full_page same recipe | `apple-crumble.md` vs `apple-crumble-wprm_print.md` | Same recipe card content, one stripped of prose |
| Same recipe, different slugs | Recipe republished under new URL | Extracted `### Ingredients` section hash |
| Cross-domain syndication | Same recipe on both sites (rare) | Ingredient list SimHash |

**Strategy**: Before ANY file is sent to the labeling API, its recipe-card hash is checked against the persistent store. If found -> blocked, no API cost incurred.

Like `LabelingClient`, `DedupStore` follows **dependency inversion**: the orchestrator imports only the abstract interface. The concrete backend (SQLite, in-memory, Redis, etc.) is injected at the composition root.

```python
# src/labeling/interfaces/dedup_store.py

class DedupStore(ABC):
    """Persistent store of recipe-card content hashes with lifecycle tracking.

    Survives process restarts. Used in sampling (dedup initial corpus)
    and during auto-labeling (block already-labeled recipes). The
    orchestrator imports ONLY this interface.
    """

    @abstractmethod
    def lookup(self, recipe_card_hash: str) -> HashStatus | None:
        """Return current hash status, or None if unknown."""
        ...

    @abstractmethod
    def register(self, recipe_card_hash: str, source_file: str,
                 status: HashStatus) -> None:
        """Persist hash with its initial lifecycle status."""
        ...

    @abstractmethod
    def update_status(self, recipe_card_hash: str, status: HashStatus) -> None:
        """Transition hash to a new status."""
        ...

    @abstractmethod
    def expire_stale(self, timeout_minutes: int) -> int:
        """Remove IN_FLIGHT entries older than timeout."""
        ...

    @abstractmethod
    def clear_in_flight_by_slugs(self, slugs: set[str]) -> None:
        """Remove IN_FLIGHT entries by source file slugs (crash recovery)."""
        ...

    @abstractmethod
    def recipe_card_hash(self, markdown: str) -> str:
        """Compute a stable fingerprint of the recipe card."""
        ...
```

```python
# src/labeling/dedup/sqlite_dedup_store.py

class SQLiteDedupStore(DedupStore):
    """Persistent dedup store backed by SQLite."""

    def __init__(self, db_path: Path):
        self._db = sqlite3.connect(str(db_path))
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS hashes ("
            "  hash TEXT PRIMARY KEY,"
            "  source_file TEXT,"
            "  status TEXT NOT NULL DEFAULT 'in_flight',"
            "  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
            ")"
        )
    ...


# src/labeling/dedup/memory_dedup_store.py

class MemoryDedupStore(DedupStore):
    """In-memory dedup store for testing. Lost on restart."""

    def __init__(self):
        self._hashes: set[str] = set()
    ...
```

**Concrete implementations**:

| Implementation | Backend | Use Case |
|---|---|---|
| `SQLiteDedupStore` | SQLite file | Production — survives restarts |
| `MemoryDedupStore` | `set[str]` in memory | Unit tests — fast, disposable |

**Storage**: SQLite file at `data/dedup_store.db`. Lightweight (~1KB per 1000 entries). The same store is used during bootstrap labeling AND production auto-labeling — preventing both duplicate API costs and duplicate training data.

Expected dedup: ~200-400 files removed from the 2,900 candidate pool.

### Three-Round Sampling (with Checkpoint)

| Round | Count | Purpose |
|-------|-------|---------|
| Round 1 | 200 | Broad coverage -> initial prompt testing |
| Round 2 | 200 | Fill gaps found in Round 1, refine prompt |
| Round 3 | 100 | Targeted edge cases, ambiguous cuisine, complex recipes |
| **Total** | **500** | Target for bootstrap |

### CheckpointManager: Resume-Safe Labeling

The labeling pipeline must survive interruptions (API failures, process kills, network issues) without losing progress. A `CheckpointManager` tracks every file through the labeling lifecycle:

```
File lifecycle states:
  pending    -> file in manifest, not yet processed
  in_flight  -> API call in progress (prevent double-submit on crash recovery)
  completed  -> labeled + validated + saved
  failed     -> exceeded retries, flagged for manual review
  skipped    -> filtered out by preprocessor or dedup
```

```python
# src/labeling/checkpoint.py

class CheckpointManager:
    """Tracks labeling progress and enables crash-safe resume.
    
    Persisted to data/checkpoint.json, atomically written after each file.
    Same checkpoint system works for bootstrap (Claude API) and production
    (local model) — the client backend is irrelevant to progress tracking.
    """

    def __init__(self, checkpoint_path: Path):
        ...

    def load(self) -> CheckpointState:
        """Restore: completed set, pending queue, in_flight set, failed list."""
        ...

    def mark_in_flight(self, slug: str) -> None:
        """Called before client.label(). Prevents double-submit on resume."""
        ...

    def mark_completed(self, slug: str, result: LabelResult) -> None:
        """Called after successful validation + save. Atomically persists."""
        ...

    def mark_failed(self, slug: str, error: str, retries_left: int) -> None:
        """Called after retry exhaustion. Flags for human review."""
        ...

    def pending_count(self) -> int: ...
    def completed_count(self) -> int: ...
    def failed_count(self) -> int: ...


# Usage in orchestrator — resume-safe loop
checkpoint = CheckpointManager(Path("data/checkpoint.json"))
state = checkpoint.load()

# Resume: skip already-completed files
for entry in manifest:
    if checkpoint.is_completed(entry.slug):
        continue  # Already labeled — free skip

    checkpoint.mark_in_flight(entry.slug)
    try:
        result = await label_one(entry, client, validator, scorer)
        checkpoint.mark_completed(entry.slug, result)
    except Exception as e:
        checkpoint.mark_failed(entry.slug, str(e))

# After crash: re-run the same command, only pending/in_flight/failed files are processed
```

**Why this matters for both phases**:
- **Bootstrap**: 500 files x $0.01-0.02/API call = $5-10. A crash at file #498 without checkpoint wastes $5. With checkpoint: resume from #499, 2 seconds.
- **Production**: 5,000+ files, local GPU inference. A power outage at file #4,500 without checkpoint wastes hours of GPU time. With checkpoint: resume from #4,501.

### Stratification Dimensions

- **Domain**: ~40% recipetineats, ~60% wellplated (matches corpus)
- **Dish type**: main_dish, baked_goods, dessert, soup, snack, salad, beverage (inferred from filename heuristics)
- **Complexity**: simple (<=5 ingredients), medium (6-12), complex (>=13)
- **Source variant**: prefer wprm_print -> fall back to full_page

Output: `data/samples/manifest.jsonl`

### Holdout Set

Before any labeling begins, randomly sample **50 files** from the deduped pool and set them aside. These files will **never** be used for training or prompt development. They serve as the final evaluation benchmark for the v1 fine-tuned model.

Why hold out before labeling:
- If these files were in the labeled set, the model might memorize similar patterns -> inflated eval scores
- The holdout is a "final exam": it measures how well the model generalizes to truly unseen recipes
- 50 files = ~2% of the usable pool -- negligible impact on training data volume

Output: `data/samples/holdout.jsonl`

## Architecture: Abstract LabelingClient Interface

This is the central architectural decision. The pipeline orchestrator depends **only** on an abstract interface, never on a concrete model backend. This allows swapping Claude API (bootstrap) for the local fine-tuned model (production) with **zero code changes** in the pipeline.

```
                    LabelingClient (abstract)
                    |
                    |  label(markdown: str) -> ExtractionResult
                    |
        +-----------+-----------+
        |                       |
OpenAILabelingClient    LocalModelLabelingClient
        |                       |
   OpenAI-compatible       vLLM / transformers
        API              (production auto-labeling)
  (bootstrap phase)


                    DedupStore (abstract)
                    |
                    |  is_duplicate(hash) -> bool
                    |  register(hash, file, status)
                    |  recipe_card_hash(md) -> str
                    |
        +-----------+-----------+
        |                       |
  SQLiteDedupStore        MemoryDedupStore
        |                       |
     SQLite file            set[str]
   (production)            (testing)
```

### Interface Definition

```python
# src/labeling/interfaces.py

class LabelingClient(ABC):
    """Abstract interface for structured recipe extraction.
    
    The orchestrator imports ONLY this interface. Concrete backends are
    injected at the composition root (run.py / cli.py).
    """

    @abstractmethod
    async def label(self, markdown: str) -> ExtractionResult:
        """Extract structured food KG data from recipe markdown.
        
        Args:
            markdown: Clean recipe Markdown text.
        
        Returns:
            ExtractionResult with parsed JSON, raw response, and token usage.
        """
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable identifier for logging and dataset provenance."""
        ...
```

### Concrete Implementations

| Implementation | Backend | Phase | Notes |
|---|---|---|---|
| `OpenAILabelingClient` | OpenAI-compatible API | Bootstrap | Uses `openai` SDK, any OpenAI-format endpoint |
| `LocalModelLabelingClient` | vLLM / transformers | Production | Local GPU inference, same prompt format |

```python
# src/labeling/clients/openai_client.py

class OpenAILabelingClient(LabelingClient):
    def __init__(self, api_key: str, base_url: str | None = None,
                 model: str = "gpt-4o", prompt_builder: PromptBuilder,
                 max_concurrent: int = 5):
        self._client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,  # None = default OpenAI; set for proxies
        )
        self._model = model
        self._prompt = prompt_builder
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def label(self, markdown: str) -> ExtractionResult:
        async with self._semaphore:
            messages = self._prompt.build_messages(markdown)
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                max_tokens=4096,
                response_format={"type": "json_object"},
            )
            return ExtractionResult(
                raw_output=response.choices[0].message.content,
                model=self.model_name,
                token_usage={"input": response.usage.prompt_tokens,
                             "output": response.usage.completion_tokens},
            )

    @property
    def model_name(self) -> str:
        return self._model


# src/labeling/clients/local_client.py

class LocalModelLabelingClient(LabelingClient):
    def __init__(self, model_path: str, prompt_builder: PromptBuilder,
                 device: str = "cuda"):
        # Load fine-tuned Qwen2.5-3B QLoRA adapter
        self._model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map=device, load_in_4bit=True)
        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        self._prompt = prompt_builder

    async def label(self, markdown: str) -> ExtractionResult:
        messages = self._prompt.build_messages(markdown)
        text = self._tokenizer.apply_chat_template(messages, tokenize=False)
        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)
        outputs = self._model.generate(**inputs, max_new_tokens=4096)
        raw = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        return ExtractionResult(raw_output=raw, model=self.model_name)

    @property
    def model_name(self) -> str:
        return f"local:{Path(self._model_path).name}"
```

### Why This Matters

| Without abstraction | With abstraction |
|---|---|
| Orchestrator imports `anthropic` directly | Orchestrator imports only `LabelingClient` |
| Switching to local model = rewrite orchestrator | Swap one line at composition root |
| Cannot reuse prompt/validation/dataset pipeline | Full reuse: same prompt, same validator, same output format |
| Claude-specific error handling in business logic | Errors are normalized at the interface boundary |

## Pipeline Architecture

```
src/labeling/
├── __init__.py
├── interfaces.py               # All abstract interfaces
│   ├── LabelingClient           #   label(md) -> ExtractionResult
│   ├── DedupStore               #   is_duplicate / register / recipe_card_hash
│   └── PromptBuilder            #   build_messages(md) -> list[dict]
│
├── models/                     # Pydantic v2 data models (split by domain)
│   ├── __init__.py             #   Re-exports all public types
│   ├── enums.py                #   All enum types (DishType, TasteProfile, Dietary, etc.)
│   ├── dish.py                 #   Dish, CookingStep, CuisineRef, RelatedDish
│   ├── ingredient.py           #   Ingredient, AmountNormalized, IngredientRelation
│   ├── extraction.py           #   ExtractionOutput, ExtractionResult, LabelResult
│   └── common.py               #   Shared types (FilePath, TokenUsage, etc.)
├── config.py                   # API keys, model selection, rate limits, paths
│
├── preprocessor.py             # Classify files -> recipe pool vs non-recipe (rules + heuristics)
├── loader.py                   # Markdown loading + comment stripping + recipe-card extraction
├── sampler.py                  # Stratified sampling from preprocessed recipe pool
│
├── dedup/                      # Content-hash dedup subpackage
│   ├── __init__.py
│   ├── sqlite_store.py         #   SQLiteDedupStore : DedupStore  (persistent)
│   └── memory_store.py         #   MemoryDedupStore : DedupStore  (testing)
│
├── clients/                    # Labeling backend subpackage
│   ├── __init__.py
│   ├── openai_client.py        #   OpenAILabelingClient : LabelingClient (bootstrap)
│   └── local_client.py         #   LocalModelLabelingClient : LabelingClient (production)
│
├── prompts/                    # Prompt engineering subpackage
│   ├── __init__.py
│   ├── builder.py              #   PromptBuilder implementation
│   ├── system.txt              #   System prompt with enum tables
│   ├── schema_spec.md          #   JSON output schema reference
│   └── few_shots.jsonl         #   4 hand-crafted few-shot examples
│
├── orchestrator.py             # Main labeling loop — imports ONLY interfaces
├── checkpoint.py               # Resume-safe progress tracking
├── validator.py                # JSON parse -> schema validate -> cross-ref check
├── scorer.py                   # Confidence scoring (0-1) per extraction
├── dataset_builder.py          # Labeled JSON -> QLoRA Alpaca-format train/val JSONL
├── review_cli.py               # Minimal CLI for human review of low-confidence labels
│
├── utils/                      # Shared utilities
│   ├── __init__.py
│   └── logger.py               #   Structured logging (file + console, per-module levels)
│
├── run.py                      # Composition root & CLI (--backend claude|local)
└── README.md
```

### Subpackage Design

Only modules that genuinely need runtime backend swapping follow the abstract-interface pattern:

```
interfaces.py
    |
    +-- LabelingClient (abstract)        # bootstrap vs production
    |       |
    |       +-- ClaudeLabelingClient     (clients/claude_client.py)
    |       +-- LocalModelLabelingClient (clients/local_client.py)
    |
    +-- DedupStore (abstract)            # SQLite vs in-memory
    |       |
    |       +-- SQLiteDedupStore         (dedup/sqlite_store.py)
    |       +-- MemoryDedupStore         (dedup/memory_store.py)
    |
    +-- PromptBuilder (abstract)         # prompt assembly strategy
            |
            +-- PromptBuilder           (prompts/builder.py)
```

`preprocessor.py` is a single concrete module — it runs once to produce the clean recipe pool, no backend-swapping needed.

### Orchestrator (model-agnostic)

```python
# src/labeling/orchestrator.py

class LabelingPipeline:
    def __init__(
        self,
        client: LabelingClient,      # <- abstract interface, NOT concrete class
        validator: OutputValidator,
        scorer: ConfidenceScorer,
    ):
        self._client = client
        self._validator = validator
        self._scorer = scorer

    async def run(self, manifest: list[SampleEntry]) -> list[LabelResult]:
        results = []
        for entry in manifest:
            markdown = load_markdown(entry.path)
            extraction = await self._client.label(markdown)
            validation = self._validator.check(extraction.parsed_output)
            confidence = self._scorer.score(extraction, validation)
            results.append(LabelResult(
                source=entry.slug,
                extraction=extraction,
                validation=validation,
                confidence=confidence,
                needs_review=confidence < 0.6,
            ))
            save_result(results[-1], f"data/labels/{entry.slug}.json")
        return results
```

### Composition Root (swap backends here)

```python
# src/labeling/run.py

# -- Bootstrap phase (OpenAI-compatible API) --
prompt = PromptBuilder(system_txt, schema_spec, few_shots)
client = OpenAILabelingClient(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url=os.environ.get("OPENAI_BASE_URL"),  # optional proxy
    model="gpt-4o",
    prompt_builder=prompt,
)
pipeline = LabelingPipeline(client, validator, scorer)
results = asyncio.run(pipeline.run(manifest))

# -- Production phase (local fine-tuned model) --
# SAME pipeline, different client:
client = LocalModelLabelingClient(
    model_path="models/qwen2.5-3b-fatbb-v1",
    prompt_builder=prompt,
)
pipeline = LabelingPipeline(client, validator, scorer)
results = asyncio.run(pipeline.run(auto_label_manifest))
```

### Output Structure

```
data/
├── preprocessor_report.json   # Preprocessor run results (file classifications)
├── dedup_store.db             # Persistent recipe-card hash DB (SQLite)
├── checkpoint.json            # Resume-safe labeling progress state
│
├── samples/
│   ├── manifest.jsonl         # 500 sampled files for labeling (3 rounds)
│   └── holdout.jsonl          # 50 files set aside before labeling — final evaluation only
├── labels/
│   ├── batch_001/             # Round 1 outputs (200 JSON files + batch manifest)
│   ├── batch_002/             # Round 2 (200 files)
│   └── batch_003/             # Round 3 (100 files)
├── reviews/
│   ├── low_confidence.jsonl   # Flagged for human review (confidence < 0.6)
│   └── reviewed.jsonl         # Corrected labels after human review
└── training/
    ├── train.jsonl             # ~425 examples (85% of labeled pool)
    ├── val.jsonl               # ~75 examples (15% of labeled pool, for hyperparameter tuning)
    └── dataset_stats.json      # Per-field coverage report
```

### Data Flow Through the Pipeline

```
corpus (5564 files)
    |
    v
[preprocessor.py]  --> data/preprocessor_report.json
    |                     2-pass: rules -> heuristics
    |                     2,900 recipe files identified
    v
[sampler.py]  --> data/dedup_store.db (initial population)
    v
[sampler.py]  --> data/dedup_store.db (initial population)
    |               data/samples/manifest.jsonl (500 files)
    |               data/samples/holdout.jsonl  (50 files)
    v
[orchestrator.py]
    |  for each file in manifest:
    |    1. loader.load_markdown()      # strip comments, extract recipe card
    |    2. dedup_store.is_duplicate() # check persistent hash DB -> skip if known
    |    3. checkpoint.mark_in_flight() # prevents double-submit on crash resume
    |    4. client.label()             # Claude API or local model (abstract interface)
    |    5. validator.check()          # JSON parse + schema + cross-ref + sanity
    |    6. scorer.score()             # 0-1 confidence
    |    7. save to data/labels/
    |    8. dedup_store.register()     # persist hash for future dedup
    |    9. checkpoint.mark_completed() # atomically advance checkpoint
    |
    v
[dataset_builder.py]  --> data/training/train.jsonl + val.jsonl
```

**Train/val/holdout distinction**:

| Set | Count | Source | Purpose |
|-----|-------|--------|---------|
| train | ~425 | Sampled + labeled | Actually train the model |
| val | ~75 | Sampled + labeled | Tune hyperparameters, monitor overfitting during training |
| holdout | ~50 | Set aside **before labeling** | Final v1 evaluation — never seen during training |

## Prompt Design

### Structure (3-part)

1. **System prompt** (`src/labeling/prompts/system.txt`):
   - Role + core rules (extract only what's present, `null` for unknown, don't guess)
   - Compact enum reference tables (3-5 lines each from schema SS10)
   - Output format: valid JSON only

2. **JSON Schema** (`src/labeling/prompts/schema_spec.md`):
   - Full extraction schema with `$comment` annotations
   - Fields marked required vs optional per schema SS8

3. **Few-shot examples** (4 hand-crafted pairs):
   - Ex 1: Simple main dish (African Chicken Curry) — core fields
   - Ex 2: Complex dish with multi-step (Pumpkin Layer Cake) — cooking_steps + ingredient_refs
   - Ex 3: Ambiguous cuisine (Teriyaki Salmon — Japanese/American) — confidence handling
   - Ex 4: Non-recipe page -> `{"dish": null, "reason": "not_a_recipe"}` — rejection behavior

### Key Prompt Decisions

- **Single-pass extraction** — one API call per file (not multi-step)
- **Enum injection** — all enum values in system prompt to prevent mismatches
- **Comment stripping** — truncate markdown after `## Life of Dozer` or before user comments section to save tokens
- **Model**: GPT-4o or compatible (structured JSON output with `response_format`)

## Key Module Details

### `loader.py`
- `load_markdown(path) -> str`: read file, strip user comments section
  - recipetineats: truncate at `## Life of Dozer` or after `### Nutrition Information:`
  - wellplated: truncate after the Notes section, before user comments start
- `get_recipe_card(md) -> str`: extract `### Ingredients` through `### Notes` section

### `validator.py`
Check chain:
1. JSON parse valid?
2. Required fields present? (`dish.name`, `ingredients[].name`, `ingredients[].amount`, `cooking_steps[]`)
3. Enum values in allowed sets? (fuzzy-match and auto-correct near misses)
4. `ingredient_refs` match `ingredients[].name`? (normalize case/plural before comparing)
5. Sanity: `total_time >= cooking_time + prep_time`, `servings > 0`
6. Retry if parse error or missing critical fields (max 2 retries, delegated to client)

### `scorer.py`
- High (>0.8): all required fields present, enums valid, refs consistent
- Medium (0.5-0.8): some optional fields missing, minor inconsistencies
- Low (<0.5): critical fields missing, cross-ref errors — **flag for human review**

## Iteration Workflow

```
Round 1 (200 files)
  -> Analyze: per-field completeness, enum distribution, confidence histogram
  -> Fix: update prompt/few-shots for systematic errors
  -> Re-label worst ~20 files if needed

Round 2 (200 files)
  -> Same analysis cycle, targeted gap-filling

Round 3 (100 files)
  -> Edge cases: multi-dish pages, ambiguous cuisine, dietary variants

After all rounds:
  -> Flag <0.6 confidence (~10-15% of files) for human review via review_cli.py
  -> Human corrects ~50-75 worst cases
  -> Build QLoRA dataset
```

## QLoRA Training Format

Each example as Alpaca-style instruction:
```json
{
  "instruction": "Extract structured food knowledge from the following recipe. Output valid JSON matching the FatBB food knowledge graph schema.",
  "input": "<full markdown content>",
  "output": "<ExtractionOutput JSON string>"
}
```

Wrapped in Qwen chat template:
```
<|im_start|>system
You are a food knowledge extraction specialist...<|im_end|>
<|im_start|>user
{input}<|im_end|>
<|im_start|>assistant
{output}<|im_end|>
```

Config: `r=16, alpha=32, 4-bit, batch=1, grad_accum=4, lr=2e-4, epochs=3, max_seq=8192` — fits RTX 4060 8GB.

## Edge Cases

| Case | Handling |
|------|----------|
| Multi-recipe pages ("10 Chinese dishes") | Extract first complete recipe; flag as multi_recipe |
| Two sites have different formatting | Separate comment-strip heuristics by domain |
| Ingredient amount unparseable ("to taste", "a pinch") | Keep `amount` string; `amount_normalized: null` |
| ingredient_refs mismatch ingredient names | Validator fuzzy-match (normalize case, plurals, whitespace) |
| Very long markdown (intro + recipe + comments) | Truncate intro to 2000 chars before `### Ingredients` |
| Recipe lacking cuisine info entirely | Output `cuisine: null` or low-confidence inferred cuisine |

## Verification

1. `preprocessor.py` -> scan all 5,564 files -> generates `preprocessor_report.json` with recipe/non-recipe breakdown
2. `dedup_store.py` -> run on preprocessed pool -> verify ~200-400 duplicates caught by content hash
3. `sampler.py` -> from deduped recipe pool -> stratified 500 files + 50 holdout -> check coverage across dish types, domains, complexity
4. `loader.py` -> test on 10 files from each domain, verify comment stripping and recipe-card extraction
5. Pilot: run `OpenAILabelingClient` on 5 test files -> manually inspect JSON output quality
6. `validator.py` -> run on all labeled outputs, target >=80% pass rate per batch
7. `checkpoint.py` -> simulate crash at file #30 of 50 -> resume -> confirm exactly 20 remaining files processed, 0 duplicates
8. Swap test: create `LocalModelLabelingClient` with a dummy model, confirm orchestrator runs unchanged
9. `dataset_builder.py` -> verify train.jsonl/val.jsonl loadable by `Dataset.from_json()`
10. End-to-end: preprocess -> dedup -> sample -> pilot -> 200x3 rounds with checkpoint -> review -> dataset ready
