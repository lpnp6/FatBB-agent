# Embedding Concurrency Architecture

> Version: v1.1
> Date: 2026-08-01
> Target: Concurrency model for Ollama batch embedding — architecture, tuning, and design rationale

---

## 1. Overview

`OllamaEmbeddingClient.a_batch_embedding()` generates vectors for an arbitrary
number of texts.  It splits the input into fixed-size sub-batches and processes
them concurrently, using a single concurrency limit to protect both local
resources and the Ollama server.

### Key goals

| Goal | Mechanism |
|------|-----------|
| Bound in-flight Ollama requests | `asyncio.Semaphore` — the sole concurrency gate |
| Bound OS thread count | `ThreadPoolExecutor` — sized to the same limit |
| Zero explicit locks | Single-consumer result queue |
| Per-sub-batch progress | Result queue → collector calls `on_progress` |
| Single knob | `_MAX_CONCURRENT_REQUESTS` drives semaphore + thread pool |

---

## 2. Architecture

```
                         ┌──────────────┐
                         │  Caller      │
                         │  (N texts)   │
                         └──────┬───────┘
                                │ a_batch_embedding(texts, on_progress=…)
                                ▼
                    ┌───────────────────────┐
                    │    Split into         │
                    │    sub-batches (128)  │
                    └───────────┬───────────┘
                                │
                                ▼
              ┌─────────────────────────────────┐
              │         task_queue               │
              │  (idx, batch, attempt) tuples    │
              └────────┬───────────────┬─────────┘
                       │               │
                 pull  │         pull  │
                       ▼               ▼
              ┌─────────────┐  ┌─────────────┐
              │  Worker 0   │  │  Worker 1   │  … one per sub-batch
              └──────┬──────┘  └──────┬──────┘
                     │                │
                     │  async with semaphore (limit: _MAX_CONCURRENT_REQUESTS)
                     │                │
                     ▼                ▼
              ┌─────────────────────────────────┐
              │    ThreadPoolExecutor            │
              │    (sized to _MAX_CONCURRENT_REQUESTS)      │
              │                                 │
              │  _batch_request() in OS threads  │
              │  → urllib.urlopen() HTTP POST    │
              │  → Ollama /api/embed             │
              └────────────┬────────────────────┘
                           │ result / exception
                           ▼
              ┌─────────────────────────────────┐
              │         result_queue             │
              │  {status, result, count} dicts   │
              └────────────┬────────────────────┘
                           │
                           ▼
              ┌─────────────────────────────────┐
              │       Collector (main task)      │
              │                                 │
              │  ordered[idx] = result           │
              │  completed += count              │
              │  on_progress(msg, completed, N)  │
              └─────────────────────────────────┘
```

### 2.1 Data flow

1. **Split**: input texts partitioned into sub-batches of `_BATCH_SIZE` (128).
2. **Enqueue**: each sub-batch pushed to `task_queue` as `(idx, batch, attempt=0)`.
3. **Worker pull**: N worker coroutines loop, pulling one task at a time.
4. **Semaphore acquire**: worker enters `async with semaphore` — blocks if
   `_MAX_CONCURRENT_REQUESTS` requests are already in flight.
5. **Offload**: `loop.run_in_executor(executor, self._batch_request, batch)`
   sends the blocking HTTP call to the shared `ThreadPoolExecutor`.
6. **Result/retry**: on success the worker pushes `{"status": "ok", ...}` to
   `result_queue`.  On failure it either re-queues to `task_queue` (next
   attempt) or pushes `{"status": "failed", ...}` (last attempt exhausted).
7. **Collect**: a single collector coroutine drains `result_queue`, updates
   `completed`, and calls `on_progress`.  Because only one coroutine mutates
   `completed` and `failed`, no locks are required.

---

## 3. Concurrency controls — single knob

Only semaphore and thread pool are bound; workers follow sub-batch count:

```python
_MAX_CONCURRENT_REQUESTS = 4   # the only knob (matches Ollama's default)

semaphore = asyncio.Semaphore(_MAX_CONCURRENT_REQUESTS)   # gates HTTP calls
executor  = ThreadPoolExecutor(_MAX_CONCURRENT_REQUESTS)  # threads = semaphore
n_workers = len(sub_batches)    # one coroutine per task (cheap!)
```

### 3.1 One worker per sub-batch — semaphore is the only gate

``_MAX_CONCURRENT_REQUESTS`` controls the semaphore and the thread pool
size.  Worker count is **not** limited — one coroutine is created per
sub-batch so every task has a dedicated worker waiting its turn.

Coroutines are lightweight objects (~1 KB each), so even 1,000 sub-batches
cost only ~1 MB of memory.  Limiting workers to the semaphore size was an
earlier design, but it created the opposite problem: if workers were fewer
than sub-batches, retries could be stranded in the queue with no worker to
pick them up.

By spawning one worker per sub-batch we guarantee:
- Every task has a worker that *already owns it* (taken from the queue).
- The semaphore gates actual concurrency — extra workers simply block.
- When a worker re-queues a failed task, it loops back and picks it up
  (or another finished worker does).

### 3.2 Worker lifecycle

```
for each sub-batch:
    worker = asyncio.create_task(_worker())

async def _worker():
    idx, batch, attempt = task_queue.get_nowait()   # one task, one worker
    async with semaphore:                            # ← ONLY concurrency gate
        result = await loop.run_in_executor(...)
    # on success → push to result_queue
    # on failure → re-queue or push failure
```

---

## 4. Lock-free progress tracking

### 4.1 Previous approach (locking)

```python
completed_lock = threading.Lock()

def _on_sub_batch(count):
    nonlocal completed
    with completed_lock:          # ← contended across threads
        completed += count
        on_progress("...", completed, total)

# Called from asyncio.to_thread() → arbitrary OS thread
```

Problems:
- Manual lock management scattered across the coroutine and its callbacks.
- `threading.Lock` in an async context — if the callback ever `await`s inside
  the lock, it deadlocks (the lock is held by a thread, not a coroutine).
- Lock contention under high concurrency.

### 4.2 Current approach (single-consumer queue)

```python
# Worker (multiple coroutines, multiple threads)
await result_queue.put((idx, {"status": "ok", "result": result, "count": n}))

# Collector (single coroutine, serialised by event loop)
idx, entry = await result_queue.get()
completed += entry["count"]          # ← single writer, no lock
on_progress("...", completed, total) # ← serialised, no lock
```

The collector is a single `await result_queue.get()` loop — it runs on the
event loop thread and is the **only** coroutine that touches `completed`,
`ordered`, and `failed`.  Workers only write to the result queue; they never
mutate shared state.  This eliminates all explicit locks.

---

## 5. Retry semantics

Failed sub-batches are re-queued into `task_queue` with an incremented
`attempt` counter:

```
Worker pulls (idx=3, batch, attempt=0)
  → HTTP fails
  → attempt=0 < _MAX_RETRIES → put (idx=3, batch, attempt=1) back to queue

Worker pulls (idx=3, batch, attempt=1)
  → HTTP fails again
  → attempt=1 < _MAX_RETRIES → put (idx=3, batch, attempt=2) back to queue

Worker pulls (idx=3, batch, attempt=2)
  → HTTP fails again
  → attempt=2 == _MAX_RETRIES-1 → push {"status": "failed"} to result_queue
```

Key behaviour:
- **Independent retry**: each sub-batch retries independently; a slow batch
  does not delay retries of other failed batches.
- **No head-of-line blocking**: re-queued tasks go to the end of the queue;
  first-attempt tasks from other batches proceed in parallel.
- **Bounded retries**: the `attempt` field is bounded by `_MAX_RETRIES`;
  a batch that exhausts all retries is reported as a failure.

**Previous behaviour** (round-based): all sub-batches fired concurrently;
after `asyncio.gather`, all survivors were retried together in the next round.
A single slow batch held up the retry of all other failed batches.

---

## 6. Sync vs async paths

| Aspect | `batch_embedding` (sync) | `a_batch_embedding` (async) |
|--------|--------------------------|----------------------------|
| Concurrency | Serial `for` loop | Worker pool + semaphore |
| Thread pool | Not used (blocking call on caller's thread) | `ThreadPoolExecutor` |
| Progress | After each sub-batch | After each sub-batch (via result queue) |
| Retry | `_batch_with_retry` | Worker re-queues to `task_queue` |
| Lock needed | No (single thread) | No (single consumer) |

The sync path is intentionally simple — it's used for small, ad-hoc calls
where the overhead of setting up queues/workers exceeds the benefit.

---

## 7. Configuration tuning

All knobs are module-level constants near the top of
[ollama_embedding_client.py](../src/rag/client/ollama_embedding_client.py):

| Constant | Default | Guidelines |
|----------|---------|------------|
| `_BATCH_SIZE` | 128 | Tune to keep a single HTTP payload under ~3 MB. For `nomic-embed-text` (max 2048 tokens), 128 texts ≈ 1 MB. |
| `_MAX_RETRIES` | 3 | Enough to ride out transient Ollama restarts; not so many that the caller hangs indefinitely. |
| `_MAX_CONCURRENT_REQUESTS` | 4 | The only knob.  Matches Ollama's ``OLLAMA_NUM_PARALLEL`` default so requests are never queued server-side. |

### 7.1 Aligning with Ollama's server limit

Ollama's own concurrency cap is ``OLLAMA_NUM_PARALLEL``, defaulting to **4**
(automatically reduced to 1 on low-memory systems).  Set
``_MAX_CONCURRENT_REQUESTS`` no higher than this value — exceeding it means
requests wait in Ollama's internal queue, which defeats the purpose of
client-side back-pressure.

To check the effective value: ``curl http://localhost:11434/api/ps`` (look
for ``num_parallel`` per model).

### 7.2 Tuning for GPU

If Ollama is running on GPU (e.g. `nomic-embed-text` on CUDA):
- Set `_MAX_CONCURRENT_REQUESTS` to **1** — GPU embedding is already
  highly parallelised internally; concurrent requests cause VRAM contention.
- Semaphore, workers, and thread pool all become 1 — no further tuning needed.

### 7.3 Tuning for CPU-only

If Ollama runs on CPU:
- Set `_MAX_CONCURRENT_REQUESTS` to `os.cpu_count() // 2`.
- Everything else follows automatically.

---

## 8. Design evolution — lessons learned

This section records the iterative refinements that led to the current
architecture.  Each iteration fixed a concrete problem; the final design
is the accumulation of those corrections.

### 8.1 Iteration 1 — code path

```
asyncio.to_thread(self._batch_request, batch)  # unbounded threads
```

**Problem**: ``asyncio.to_thread()`` uses the event loop's default executor,
which has no bound on the number of threads.  For 100 concurrent sub-batches
this could spawn 100 OS threads, starving the process of memory and degrading
throughput through context-switch overhead.

**Fix**: Replaced with a shared ``ThreadPoolExecutor`` of fixed size.

### 8.2 Iteration 2 — round-based retry + manual lock

```
for attempt in range(_MAX_RETRIES):
    tasks = [asyncio.to_thread(...) for batch in pending]
    outcomes = await asyncio.gather(*tasks)
    # all-or-nothing: one slow batch blocks retries for all others
    completed += len(outcome)  # race: called from multiple threads
```

**Problems**:

1. **Lock on ``completed``**: progress was updated by a callback fired from
   arbitrary OS threads (via ``asyncio.to_thread``).  A ``threading.Lock``
   protected the counter, but cross-primitive locking (thread lock used from
   async code) is fragile — an ``await`` inside the lock would deadlock.
2. **Round-based retry**: after ``asyncio.gather`` returned, ALL survivors
   were re-dispatched in the next round.  A single slow batch held up
   retries for every other failed batch.
3. **Progress granularity**: ``on_progress`` only fired once per round, not
   per sub-batch.

**Fix**: Introduced task-queue + result-queue pattern with a single-consumer
collector — no locks needed (see §4).

### 8.3 Iteration 3 — two-parameter confusion

```
_MAX_CONCURRENT_REQUESTS = 5   # semaphore
_MAX_THREAD_WORKERS = 4        # thread pool
```

**Problem**: Two knobs that must satisfy ``threads ≥ semaphore``.  When
``threads=4 < semaphore=5``, a worker could acquire a semaphore slot but find
no thread available — the slot is wasted, the semaphore provides no actual
back-pressure.  Manually keeping the two in sync is unnecessary mental
overhead.

**Fix**: Removed ``_MAX_THREAD_WORKERS``.  A single parameter
``_MAX_CONCURRENT_REQUESTS`` now sizes both semaphore and thread pool.

### 8.4 Iteration 4 — artificially limited workers

```
n_workers = min(_MAX_CONCURRENT_REQUESTS, len(sub_batches))
```

**Problem**: Workers were capped at the semaphore size out of concern for
retry re-entry — that if all workers drained the queue and exited, a retry
would have no worker to pick it up.  But this reasoning was backwards: fewer
workers means *fewer* chances for a retry to be picked up.  When a worker
fails and re-queues, it loops back immediately and picks up the re-queued
task itself — no extra workers needed.  With a cap of 5 workers for 50
sub-batches, 45 tasks sit in the queue unassigned, waiting for a worker to
finish before they're even claimed.

**Fix**: ``n_workers = len(sub_batches)`` — one coroutine per task.
Coroutines are ~1 KB each; 1,000 workers costs ~1 MB.  The semaphore is the
only concurrency gate that matters.

### 8.5 Iteration 5 — exceeding Ollama's server limit

```
_MAX_CONCURRENT_REQUESTS = 5   # Ollama default is 4
```

**Problem**: ``OLLAMA_NUM_PARALLEL`` defaults to 4.  Sending 5 concurrent
requests means the 5th one queues inside Ollama, adding latency and
undermining the client-side semaphore's purpose.  The client should match
the server's limit so back-pressure is applied before requests leave the
process.

**Fix**: Set ``_MAX_CONCURRENT_REQUESTS = 4`` to match Ollama's default.
If the server is tuned higher, the client should be updated to match.

### 8.6 Summary of principles

| # | Principle | Why |
|---|-----------|-----|
| 1 | **One knob is better than two** when they always move together. | Fewer mistakes, fewer configs to audit. |
| 2 | **Let the semaphore gate everything.** Don't invent second sources of back-pressure. | Each extra limit interacts non-trivially with the others. |
| 3 | **Don't artificially cap coroutine count.** They're cheap; over-capping creates under-subscription. | The bottleneck should be I/O, not an arbitrary worker count. |
| 4 | **Match the server's concurrency model.** Client-side limits should align with server-side limits. | Otherwise back-pressure fires on the wrong side of the wire. |
| 5 | **Queues decouple producers from consumers.** A result queue + single collector eliminates locks. | "Don't communicate by sharing memory; share memory by communicating." |

---

## 9. Related files

| File | Role |
|------|------|
| [src/rag/client/ollama_embedding_client.py](../src/rag/client/ollama_embedding_client.py) | Core implementation |
| [src/rag/interfaces/client.py](../src/rag/interfaces/client.py) | `EmbeddingClient` abstract interface |
| [src/rag/stores/postgres/postgres_vector_search_store.py](../src/rag/stores/postgres/postgres_vector_search_store.py) | Primary caller of batch embedding |
| [src/fatbb/infrastructure/kb/vector.py](../src/fatbb/infrastructure/kb/vector.py) | Vector KB adapter — constructs client instances |
| [docs/windows-postgres-connection.md](windows-postgres-connection.md) | Related: connection pooling for PostgreSQL on Windows |
