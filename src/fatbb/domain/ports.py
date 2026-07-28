"""Application ports. New backends implement these rather than changing the CLI."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from rag.interfaces.indexer import Indexer
from rag.interfaces.retriever import Retriever
from rag.models.document import Document

from .knowledge_base import KnowledgeBase


class KnowledgeBaseRepository(Protocol):
    def list(self) -> list[KnowledgeBase]: ...

    def create(self, knowledge_base: KnowledgeBase) -> None: ...


class SourceImporter(Protocol):
    type: str

    def load(self, path: str, *, knowledge_base_id: str) -> Sequence[Document]: ...


class RetrievalBackend(Protocol):
    type: str

    def indexer(self) -> Indexer: ...

    def retriever(self) -> Retriever: ...
