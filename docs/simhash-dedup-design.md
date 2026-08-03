# SimHash Dedup Design

## 1. Problem

Given a recipe markdown document, determine whether it is a **near-duplicate** of
a previously processed document — the same recipe republished with different
formatting, ad blocks, or minor wording changes.

**Similarity measure**: 64-bit SimHash + Hamming distance.

- Two documents are near-duplicates when their SimHash fingerprints differ by
  ≤ *T* bits (Hamming distance, default 3).
- SimHash uses weighted voting across tokens: local edits flip only a few bits,
  so structurally similar texts produce fingerprints that stay close.

## 2. Manku Block Index — Forward Pruning

### 2.1 Core Proposition (Pigeonhole Principle)

> For a 64-bit hash and Hamming threshold *T*, partition the hash into *T*+1
> blocks. Two hashes within distance *T* MUST share at least one identical block.

Proof: if two hashes differ in all *T*+1 blocks, each block contributes at least
1 bit of difference, giving total distance ≥ *T*+1 > *T* — contradiction.

### 2.2 Index Structure

| Parameter | Value |
|---|---|
| Hash width | 64 bit |
| Threshold T | 3 |
| Block count | 4 (= T+1) |
| Block width | 16 bit |
| Bucket count | 4 × 65536 |

Each registered hash writes 4 rows into `simhash_index`:

```
(block_id=0, block_value=0x3f2a, hash="abc...")
(block_id=1, block_value=0x9e01, hash="abc...")
(block_id=2, block_value=0x74d3, hash="abc...")
(block_id=3, block_value=0x2b8f, hash="abc...")
```

### 2.3 Lookup Flow

```
lookup(query_hash):
    1. Split query_hash into 4 16-bit blocks
    2. For each block, query simhash_index to collect candidates
    3. For each candidate, perform exact Hamming distance check (secondary verification)
```

**Pruning effect**: average bucket size = N / 65536. At 100k entries → ~1.5
candidates per bucket, ~6 candidates per single lookup (4 buckets × 1.5), rather
than a full scan of 100k.

### 2.4 Why Secondary Verification Is Necessary

The block index gives a **necessary** condition, not a **sufficient** one — two
hashes sharing a 16-bit block does not guarantee full 64-bit Hamming distance ≤ 3.
Concrete example:

```
hash_a: 0000 0000 0000 0000  (all zeros)
hash_b: 0000 000f 0000 0000  (shares block 0, but block 1 differs by 4 bits → distance 4 > 3)
```

The Manku index guarantees **no false negatives** (recall = 1.0), not **no false
positives**. Secondary Hamming verification is always required.

---

## 3. Batch Lookup Optimization — Reverse Index

### 3.1 The Naive Batch Problem

```
lookup_batch_naive(hashes[n]):
    # Phase 1: 4n SQL queries
    for h in hashes:
        4 × SELECT ... WHERE block_id=? AND block_value=?

    # Phase 2: 1 SQL query (fetch statuses)

    # Phase 3: O(n·m) Hamming comparisons
    for h in hashes:                    # n
        for candidate in candidates:    # m
            hamming(h, candidate)       # 2 × int(hex, 16) per call
```

After Phase 1, the association of "which candidate shares a block with which
input" is discarded. Phase 3 degenerates into a Cartesian product.

### 3.2 Reverse Index Fix

Core idea: **let Phase 1 results point back to the specific input hashes**.

```
Precompute:
    (block_id, block_value) → {input_index_0, input_index_5, ...}

Phase 1 — batched query (max 4 queries):
    SELECT hash, block_value FROM simhash_index
    WHERE block_id=0 AND block_value IN (v₁, v₂, ..., vₙ)

    For each result row (candidate_hash, block_value):
        look up reverse index → get relevant input_indices

Phase 2 — status fetch (1 query, unchanged)

Phase 3 — targeted Hamming check:
    for candidate_hash, input_indices:      # each candidate → only its relevant inputs
        for i in input_indices:             # average ~1–2 inputs
            hamming(hashes[i], candidate)   # int pre-computed once
```

### 3.3 Complexity Comparison

n = input hash count, m = unique candidate hash count

| Phase | Naive | Optimized |
|---|---|---|
| 1. Block-index queries | 4n SQL round-trips | ≤4 SQL round-trips |
| 2. Status fetch | 1 query | 1 query |
| 3. Hamming check | n·m comparisons + 2·n·m `int(hex,16)` | ~6n comparisons (typical) + n+m `int(hex,16)` |

Concrete scenario n=200, m=500:
- Phase 1: 800 → 4 round-trips
- Phase 3: 100,000 → ~1,200 Hamming comparisons (~83× reduction)

### 3.4 Isomorphism with the Manku Index

The Manku index uses blocks as a **forward index**: `(block_id, block_value) →
candidate hashes`, avoiding a full table scan.

The reverse index uses the **same blocks** as a **reverse index**: `(block_id,
block_value) → input indices`, avoiding a full candidate comparison.

Both prune the same category of redundancy — the Manku guarantee that two
related hashes must share a block becomes the scheduling primitive for
computation: only `(query, candidate)` pairs that actually share a block undergo
Hamming comparison. Pairs that the Manku index prunes at lookup time are never
resurrected in Phase 3.

---

## 4. References

- Manku, Jain, Sarma. "Detecting Near-Duplicates for Web Crawling." WWW 2007.
- Charikar. "Similarity Estimation Techniques from Rounding Algorithms." STOC 2002. (SimHash original paper)
