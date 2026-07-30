"""Use cases for creating, selecting, indexing, and querying knowledge bases."""

from __future__ import annotations

from collections.abc import Callable
import logging
from uuid import uuid4

from rag.models.query import RetrievalQuery
from rag.models.evidence import Evidence

from fatbb.application.registry import CapabilityRegistry
from fatbb.domain.knowledge_base import KnowledgeBase, KnowledgeBaseConfig
from fatbb.domain.ports import KnowledgeBaseRepository


logger = logging.getLogger(__name__)


class KnowledgeBaseService:
    """Coordinate knowledge-base persistence, ingestion, and retrieval.

    Knowledge-base configuration is persisted locally, while each knowledge
    base supplies the database URL used by its retrieval adapter.
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
        source_type: str, source_path: str, *,
        on_progress: Callable[[str, int, int], None] | None = None,
    ) -> KnowledgeBase:
        """Import a source path, index its documents, and persist its metadata.

        Documents are indexed before the knowledge-base record is saved, so an
        empty or unsupported source never produces a selectable empty entry.
        """
        normalized_name = name.strip()
        if not normalized_name:
            raise ValueError("Knowledge base name cannot be empty.")
        if not database_url.strip():
            raise ValueError("Database URL cannot be empty.")

        # Capture adapter choices and the per-knowledge-base database URL.
        config = KnowledgeBaseConfig(
            retrieval_type=retrieval_type, database_type=database_type,
            database_url=database_url.strip(), source_type=source_type,
        )
        knowledge_base = KnowledgeBase(
            id=str(uuid4()), name=normalized_name, config=config, source_path=source_path
        )

        logger.info(
            "Resolving knowledge-base adapter for creation",
            extra={"retrieval_type": config.retrieval_type, "database_type": config.database_type},
        )
        adapter = self._registry.knowledge_base(config.retrieval_type, config.database_type)
        if on_progress is not None:
            on_progress("Connecting to database", 0, 1)
        try:
            adapter.check_connection(config.database_url)
            if on_progress is not None:
                on_progress("Database connection successful", 1, 1)
        except Exception as error:
            logger.exception(
                "Database connection check failed",
                extra={"knowledge_base": knowledge_base.name, "database_type": config.database_type},
            )
            raise ValueError(
                f"Could not connect to the configured {config.database_type} database."
            ) from error

        # Importers attach the generated ID to each document for retrieval scoping.
        logger.info(
            "Resolving importer adapter for source_type=%r",
            config.source_type,
        )
        logger.info("Loading source documents", extra={"knowledge_base": knowledge_base.name})
        importer = self._registry.importer(config.source_type)
        documents = importer.load(source_path, knowledge_base_id=knowledge_base.id, on_progress=on_progress)
        if not documents:
            raise ValueError("No supported files were found at the specified path.")

        # Index first; persist the entry only after its documents are searchable.
        logger.info(
            "Indexing source documents",
            extra={"knowledge_base": knowledge_base.name, "document_count": len(documents)},
        )
        adapter.indexer(config.database_url).upsert_documents(documents, on_progress=on_progress)
        self._repository.create_knowledge_base(knowledge_base)
        logger.info(
            "Knowledge base created",
            extra={"knowledge_base": knowledge_base.name, "document_count": len(documents)},
        )
        return knowledge_base

    def retrieve(self, knowledge_base: KnowledgeBase, question: str) -> list[Evidence]:
        """Retrieve the five best keyword matches scoped to one knowledge base."""
        logger.info(
            "Resolving knowledge-base adapter for retrieval",
            extra={"retrieval_type": knowledge_base.config.retrieval_type,
                   "database_type": knowledge_base.config.database_type},
        )
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
