# Windows PostgreSQL Connection Performance

> Version: v1.1
> Date: 2026-07-31
> Target: Document the root causes and fixes for slow first-connection latency on Windows

---

## 1. Symptoms

On Windows, database operations felt slow because every query opened a fresh
TCP connection. The first operation also paid a DNS resolution penalty when
`localhost` was used in the connection string.

---

## 2. Root Causes

Two independent issues compound:

| # | Cause | Mechanism | Latency | Windows-specific? |
|---|-------|-----------|---------|-------------------|
| ① | `localhost` → `::1` fallback | DNS resolves to IPv6 first; PostgreSQL is not bound to `::1`; TCP SYN times out, then IPv4 is tried | 1-3 s | Yes (macOS has fast dual-stack; Linux `/etc/hosts` short-circuits) |
| ② | No connection pooling | `psycopg.connect()` is called per-operation, paying TCP + auth cost every time | 50-500 ms per call | No (but cumulative impact is worse on Windows) |

### 2.1 Issue ① — IPv6-to-IPv4 fallback

```
Client (Windows)                          Server (PostgreSQL)
     │                                           │
     │  getaddrinfo("localhost")                 │
     │  → [::1]:5432, [127.0.0.1]:5432           │
     │                                           │
     │──TCP SYN → ::1:5432──────────────────────►│  (not listening on IPv6)
     │     ... timeout 1-3 s ...                  │
     │──TCP SYN → 127.0.0.1:5432────────────────►│  ✓ connected
```

On macOS, PostgreSQL installed via Homebrew binds to `*` (all interfaces,
including IPv6), so `::1` succeeds immediately. On Windows, the default
PostgreSQL installer (or Docker with port mapping `0.0.0.0:5432`) only binds
to IPv4.

### 2.2 Issue ② — No connection pooling

Every database operation opened a fresh `psycopg.connect()` and closed it
immediately. On Windows each new TCP connection incurs ~20-50 ms of overhead
(firewall inspection, TCP handshake, backend fork). A single user action
triggering 3-5 operations accumulated 100-500 ms of connection overhead.

---

## 3. Implemented Fixes

All fixes are in [src/rag/stores/postgres/postgres_bm25_search_store.py](../src/rag/stores/postgres/postgres_bm25_search_store.py).

### 3.1 Connection pooling (`_get_pool`)

A module-level `ConnectionPool` is created per DSN and reused across all store
instances:

```python
_pools: dict[str, psycopg_pool.ConnectionPool] = {}

def _get_pool(dsn: str):
    dsn = _normalize_dsn(dsn)
    if dsn not in _pools:
        import psycopg_pool
        _pools[dsn] = psycopg_pool.ConnectionPool(
            dsn,
            min_size=1,
            max_size=10,
            kwargs={"connect_timeout": _DEFAULT_CONNECT_TIMEOUT},
        )
    return _pools[dsn]
```

- `min_size=1` — one connection is pre-warmed so the first `.connection()`
  borrow is instant (after pool init).
- `max_size=10` — enough headroom for concurrent indexing bursts without
  overwhelming the database.
- `connect_timeout=10` — TCP handshake timeout; fails fast instead of hanging.
- `_connect()` changed from `psycopg.connect(dsn)` to
  `_get_pool(dsn).connection(timeout=15)`. All 8 call sites are unaffected
  because `pool.connection()` is also a context manager.

### 3.2 DSN normalization (`_normalize_dsn`)

On Windows, `localhost` in the connection string is replaced with `127.0.0.1`
before the pool is created. This bypasses the DNS resolver entirely.

```python
def _normalize_dsn(dsn: str) -> str:
    if os.name != "nt":
        return dsn
    dsn = re.sub(r"(?<=@)localhost(?=[:/]|$)", "127.0.0.1", dsn)  # URI format
    dsn = re.sub(r"\bhost=localhost\b", "host=127.0.0.1", dsn)     # key=value format
    return dsn
```

---

## 4. Design Decisions

| Decision | Rationale |
|----------|-----------|
| Pool lifetime = module-level | `PostgresBM25SearchStore` is created per-operation in the KB adapters; a pool per instance would defeat the purpose. Module-level cache ensures one pool per DSN for the process lifetime. |
| `_normalize_dsn` before pool key lookup | If a user types `localhost` and another types `127.0.0.1` for the same server, two separate pools would be created. Normalizing before the key lookup merges them into one. |
| No DSN normalization on non-Windows | On macOS/Linux, `localhost` resolution is fast and PostgreSQL commonly listens on IPv6. Normalizing would be an unnecessary transformation. |
| `pool.connection(timeout=15)` | Aligned with `connect_timeout=10` plus buffer; prevents the default 30 s wait when the database is unreachable. |

---

## 5. Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `psycopg[binary,pool]` | ≥ 3.2, < 4 | PostgreSQL driver + `ConnectionPool` |
| `psycopg_pool` | (pulled by extra) | Connection pooling |

---

## 6. Related Files

| File | Role |
|------|------|
| [src/rag/stores/postgres/postgres_bm25_search_store.py](../src/rag/stores/postgres/postgres_bm25_search_store.py) | Core: pool, normalization, connection factory |
| [src/rag/stores/postgres/postgres_vector_search_store.py](../src/rag/stores/postgres/postgres_vector_search_store.py) | Inherits `_connect()` — benefits automatically |
| [src/fatbb/infrastructure/kb/bm25.py](../src/fatbb/infrastructure/kb/bm25.py) | BM25 adapter — creates store instances |
| [src/fatbb/infrastructure/kb/vector.py](../src/fatbb/infrastructure/kb/vector.py) | Vector adapter — creates store instances |
| [requirements.txt](../requirements.txt) | `psycopg[binary,pool]` dependency declaration |
