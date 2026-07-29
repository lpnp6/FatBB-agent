"""Use cases for creating, selecting, indexing, and querying knowledge bases."""

from __future__ import annotations

from uuid import uuid4

from rag.models.query import RetrievalQuery
from rag.models.evidence import Evidence

from fatbb.application.registry import CapabilityRegistry
from fatbb.domain.knowledge_base import KnowledgeBase, KnowledgeBaseConfig
from fatbb.domain.ports import KnowledgeBaseRepository


class KnowledgeBaseService:
    """Coordinate knowledge-base persistence, ingestion, and retrieval.

    Knowledge-base configuration is persisted locally, while each knowledge
    base supplies the PostgreSQL URL used by its retrieval adapter.
    """

    def __init__(self, repository: KnowledgeBaseRepository, registry: CapabilityRegistry):
        """Store the local catalog repository and registered capabilities."""
        self._repository = repository
        self._registry = registry

    def list(self) -> list[KnowledgeBase]:
        """List knowledge bases from the local CLI catalog."""
        return self._repository.list_knowledge_bases()

    def create(
        self, name: str, retrieval_type: str, database_type: str, database_url: str,
        source_type: str, source_path: str,
    ) -> KnowledgeBase:
        """Import a source path, index its documents, and persist its metadata.

        Documents are indexed before the knowledge-base record is saved, so an
        empty or unsupported source never produces a selectable empty entry.
        """
        normalized_name = name.strip()
        if not normalized_name:
            raise ValueError("Knowledge base name cannot be empty.")
        if not database_url.strip():
            raise ValueError("PostgreSQL URL cannot be empty.")

        # Capture adapter choices and the per-knowledge-base database URL.
        config = KnowledgeBaseConfig(
            retrieval_type=retrieval_type, database_type=database_type,
            database_url=database_url.strip(), source_type=source_type,
        )
        knowledge_base = KnowledgeBase(
            id=str(uuid4()), name=normalized_name, config=config, source_path=source_path
        )

        # Importers attach the generated ID to each document for retrieval scoping.
        importer = self._registry.importer(config.source_type)
        documents = importer.load(source_path, knowledge_base_id=knowledge_base.id)
        if not documents:
            raise ValueError("No supported files were found at the specified path.")

        # Index first; persist the entry only after its documents are searchable.
        self._registry.knowledge_base(config.retrieval_type, config.database_type).indexer(
            config.database_url
        ).upsert_documents(documents)
        self._repository.create_knowledge_base(knowledge_base)
        return knowledge_base

    def retrieve(self, knowledge_base: KnowledgeBase, question: str) -> list[Evidence]:
        """Retrieve the five best keyword matches scoped to one knowledge base."""
        return self._registry.knowledge_base(
            knowledge_base.config.retrieval_type, knowledge_base.config.database_type
        ).retriever(
            knowledge_base.config.database_url
        ).retrieve(
            RetrievalQuery(
                text=question,
                # Keyword retrieval is the initial supported query strategy.
                mode="keyword",
                top_k=5,
                # Never allow a query to return documents from another base.
                filters={"knowledge_base_id": knowledge_base.id},
            )
        )
