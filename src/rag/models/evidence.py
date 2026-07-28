"""Uniform, citeable output of a retrieval operation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .common import Metadata, SourceRef
from .document import TextChunk

EvidenceKind = Literal["text_chunk", "graph_node", "graph_edge", "graph_path"]


@dataclass(frozen=True)
class Evidence:
    """A scored fact that can be included in an LLM context window."""

    id: str
    kind: EvidenceKind
    content: str
    score: float
    source: SourceRef | None = None
    metadata: Metadata = field(default_factory=dict)
    chunk: TextChunk | None = None

    def __post_init__(self) -> None:
        if self.kind == "text_chunk" and self.chunk is None:
            raise ValueError("text_chunk evidence requires a chunk")
        if self.kind != "text_chunk" and self.chunk is not None:
            raise ValueError("only text_chunk evidence may contain a chunk")
