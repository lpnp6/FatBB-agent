# Distributed Labeling — Result Channel Design

## 1. Problem

The distributed labeling pipeline (`src/labeling/pipeline/distribute/`) currently
runs workers on remote machines, but each worker writes dedup/checkpoint state to
its **own local** SQLite/JSON files. The orchestrator's copy and every worker's
copy diverge, so:

- dedup decisions (near-duplicate detection) are not shared across workers;
- checkpoint status (PENDING / IN_FLIGHT / ACCEPTED) is not authoritative;
- the local data cannot simply be migrated to Redis (option A was rejected for
  this reason — all existing data lives in local files).

**Goal**: keep the dedup/checkpoint data local to the orchestrator (single
writer), while making workers fully stateless — they only label and ship results
back over Redis.

## 2. Architecture — two Redis Stream channels

```
现在:
  orchestrator ──enqueue──▶ Redis tasks ──dequeue──▶ worker ──写本地 SQLite/JSON (分裂)──▶ ✗

方案 B:
  orchestrator ──enqueue──▶ Redis tasks ──dequeue──▶ worker ──标注──┐
                                                                   │
  orchestrator ◀──consume── Redis results ◀──publish──◀────────────┘
       │
       └── 写本地 dedup/checkpoint (单写者)
```

- `labeling:tasks` (existing) — orchestrator → workers. Consumer group
  `labeling-workers`.
- `labeling:results` (new) — workers → orchestrator. Consumer group
  `labeling-results` (single consumer: the orchestrator).

The queue becomes a **pure transport layer**: it owns no dedup/checkpoint state.
The orchestrator is the sole writer of that state, persisting it as results
arrive.

## 3. Result message schema

The worker publishes a plain dict (no `DedupEntry` / store types on the worker).
The orchestrator consumes it and reconstructs the store write.

```json
// success
{
  "source_id": "...",
  "recipe_card_hash": "...",
  "outcome": "success",
  "raw_text": "...",
  "model": "...",
  "output": "{...}"
}
```
```json
// failure — raw_text is carried so the orchestrator can re-enqueue without
// re-resolving the source file
{
  "source_id": "...",
  "recipe_card_hash": "...",
  "raw_text": "...",
  "outcome": "failure",
  "last_error": "..."
}
```

The task's `_message_id` (set at dequeue time) is carried alongside the result in
`publish_results` so the queue can `XACK` the task in the same transaction that
publishes the result — a result is never lost before its task is acknowledged.

## 4. WorkQueue interface (new contract)

```python
class WorkQueue(ABC):
    async def load(self) -> None                              # 建两个消费组
    async def enqueue(self, items: list[dict]) -> None        # XADD tasks
    async def dequeue(self, count) -> list[dict]              # XREADGROUP tasks（内含 reclaim_stale）
    async def publish_results(self, results: list[tuple[task, result_dict]]) -> None
                                                              # 一个 MULTI：N×XADD result + N×XACK task
    async def consume_results(self, count) -> list[dict]      # XREADGROUP results
    async def reclaim_stale(self) -> int                      # XAUTOCLAIM tasks
    async def reclaim_stale_results(self) -> int              # XAUTOCLAIM results
```

Removed: `join`, `submit_results`, `submit_retries`, the `outstanding` set, and
the `dedup_store` / `checkpoint` constructor args.

## 5. Per-file changes

| File | Change |
|---|---|
| `interfaces/work_queue.py` | New contract (section 4) |
| `queue/redis_streams.py` | Add `results` stream/group; remove `outstanding`/`join`/`submit_*`; implement `publish_results` / `consume_results` / `reclaim_stale_results`; drop `dedup_store` / `checkpoint` ctor args |
| `pipeline/distribute/worker.py` | Drop `DedupEntry`/`HashStatus` imports; `_process_one` returns a plain `result_dict`; success and failure both go through `publish_results`; no store writes |
| `pipeline/distribute/orchestrator.py` | After `enqueue`, run a drain loop: `consume_results` → success ⇒ `register_batch` + `mark_completed_batch`; failure ⇒ `mark_pending` + `mark_in_flight` + `enqueue` (retry); local `active` set is the barrier |
| `pipeline/distribute/run_default.py` | Inject stores into the orchestrator only; the worker gets queue + client + validator |
| `tests/test_redis_streams_queue.py` | Test publish/consume semantics |

## 6. Key semantics

### 6.1 Barrier (replaces `join` + `outstanding`)

The orchestrator enqueues a batch of N, then tracks a local
`active = {source_id, ...}`. Each consumed success result discards its id; each
failure is re-enqueued and **stays** in `active`. The next batch is enqueued only
when `active` is empty. Single-writer, purely local — simpler than a Redis set.

### 6.2 Retry

The orchestrator owns the retry decision. On a failure result it
`mark_pending` + `mark_in_flight` + re-`enqueue` the task (with `last_error`).
Retry limit = **2** (3 total attempts), matching the existing non-distributed
orchestrator's `--retries` default; configurable via the orchestrator
constructor.

### 6.3 Crash recovery

- worker crashes → task message stays pending → existing `reclaim_stale`
  (XAUTOCLAIM) redelivers it.
- orchestrator crashes after a result was published but before it was consumed →
  result message stays pending → `reclaim_stale_results` (XAUTOCLAIM) recovers it.
- orchestrator crashes between reading (ack-on-read) and persisting a result →
  that result is dropped for this run, but its checkpoint item is still
  `IN_FLIGHT` and is re-picked on the next run (same recovery as the
  non-distributed orchestrator). No permanent loss.
- duplicate results: the orchestrator registered `IN_FLIGHT` at enqueue time, so
  a repeated success is just overwritten to `ACCEPTED` — idempotent.

## 7. Decisions (resolved)

1. **Result channel consumption**: consumer group + XAUTOCLAIM (recommended),
   symmetric with the task stream.
2. **Retry limit**: `retries=2`, orchestrator ctor param.
3. **`outstanding` set**: deleted entirely — the barrier is orchestrator-local.

## 8. Implementation order

1. `work_queue.py` interface → 2. `redis_streams.py` impl + tests → 3.
   `worker.py` → 4. `orchestrator.py` → 5. `run_default.py` injection.
