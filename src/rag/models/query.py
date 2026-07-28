"""Input model for all retrievers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .common import Metadata

RetrievalMode = Literal["keyword", "vector", "graph", "hybrid"]


@dataclass(frozen=True)
class RetrievalQuery:
    """A normalized request for knowledge retrieval."""

    text: str
    top_k: int = 5
    mode: RetrievalMode = "hybrid"
    filters: Metadata = field(default_factory=dict)
    entity_ids: tuple[str, ...] = ()
    relation_types: tuple[str, ...] = ()
    max_hops: int = 2

    def __post_init__(self) -> None:
        if self.top_k < 1:
            raise ValueError("top_k must be at least 1")
        if self.max_hops < 0:
            raise ValueError("max_hops cannot be negative")
