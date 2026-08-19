"""Reciprocal Rank Fusion: merge score-ordered id lists without comparable scales."""

from __future__ import annotations

from collections.abc import Iterable, Sequence


def rrf(ranked_lists: Iterable[Sequence[str]], *, k: int = 60) -> dict[str, float]:
    """Fuse ranked id lists into one relevance dict.

    Each list is assumed score-descending (highest first). Only rank position
    matters — Lucene relevance, cosine, and BM25 scores need never be
    comparable. A candidate present in several lists accumulates
    ``sum(1 / (k + rank))``, so one seen by multiple channels beats one seen
    once, and within a channel order is preserved.
    """
    fused: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked):
            fused[item] = fused.get(item, 0.0) + 1.0 / (k + rank)
    return fused
