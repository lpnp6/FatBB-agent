"""Source documents and their text-search units."""

from __future__ import annotations

from dataclasses import dataclass, field

from .common import Metadata, SourceRef


@dataclass(frozen=True)
class Document:
    """Original, unchunked knowledge source."""

    id: str
    content: str
    source: SourceRef
    metadata: Metadata = field(default_factory=dict)


@dataclass(frozen=True)
class TextChunk:
    """The minimum text unit used by lexical and vector retrieval."""

    id: str
    document_id: str
    content: str
    index: int
    source: SourceRef
    start_offset: int | None = None
    end_offset: int | None = None
    metadata: Metadata = field(default_factory=dict)


@dataclass(frozen=True)
class ScoredTextChunk:
    """A text chunk and the relevance score computed by a search backend."""

    chunk: TextChunk
    score: float
