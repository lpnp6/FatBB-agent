"""Application ports. New knowledge-base adapters extend the CLI."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from rag.interfaces.indexer import Indexer
from rag.interfaces.retriever import Retriever
from rag.models.document import Document

from .knowledge_base import KnowledgeBase


class KnowledgeBaseRepository(Protocol):
    """Port for persisting knowledge-base configuration, not indexed chunks.

    Implementations own configuration storage only. Chunk persistence remains
    behind the existing RAG store interfaces, keeping configuration lifecycle
    separate from document indexing.
    """

    def list_knowledge_bases(self) -> list[KnowledgeBase]: ...

    def create_knowledge_base(self, knowledge_base: KnowledgeBase) -> None: ...


class SourceImporter(Protocol):
    """Port that converts one configured source into RAG ``Document`` values.

    ``type`` is the registry key stored in ``KnowledgeBaseConfig``. Importers
    must attach ``knowledge_base_id`` to every document's metadata so all
    descendant chunks inherit the isolation boundary used at query time.
    """

    # Registry key, for example ``file_path`` or a future ``s3`` importer.
    type: str

    def load(self, path: str, *, knowledge_base_id: str) -> Sequence[Document]: ...


class KnowledgeBaseAdapter(Protocol):
    """Port that supplies compatible index and retrieval operations for one KB.

    An implementation may compose any store or retrieval strategy, provided
    its indexer and retriever conform to the existing RAG interfaces. This is
    the extension seam for future vector, graph, or hybrid knowledge bases.
    """

    # Capability key, for example ``bm25`` or a future ``vector`` KB.
    type: str

    def check_connection(self, database_url: str) -> None:
        """Raise when this adapter cannot connect to its configured database."""

    def indexer(self, database_url: str) -> Indexer: ...

    def retriever(self, database_url: str) -> Retriever: ...
