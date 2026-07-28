"""Common value objects shared by RAG models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

Metadata: TypeAlias = dict[str, object]


@dataclass(frozen=True)
class SourceRef:
    """A stable, human-inspectable pointer to the origin of knowledge."""

    document_id: str | None = None
    uri: str | None = None
    title: str | None = None
    locator: str | None = None
