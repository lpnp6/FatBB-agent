"""Knowledge-base configuration independent of a concrete backend."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KnowledgeBaseConfig:
    retrieval_type: str = "bm25"
    database_type: str = "pg"
    source_type: str = "file_path"


@dataclass(frozen=True)
class KnowledgeBase:
    id: str
    name: str
    config: KnowledgeBaseConfig
    source_path: str
