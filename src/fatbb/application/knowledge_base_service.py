"""Use cases for creating, selecting, indexing, and querying knowledge bases."""

from __future__ import annotations

from dataclasses import replace
from uuid import uuid4

from rag.models.query import RetrievalQuery
from rag.models.evidence import Evidence

from fatbb.application.registry import CapabilityRegistry
from fatbb.domain.knowledge_base import KnowledgeBase, KnowledgeBaseConfig
from fatbb.domain.ports import KnowledgeBaseRepository


class KnowledgeBaseService:
    def __init__(self, repository: KnowledgeBaseRepository, registry: CapabilityRegistry):
        self._repository = repository
        self._registry = registry

    def list(self) -> list[KnowledgeBase]:
        return self._repository.list()

    def create(self, name: str, source_path: str) -> KnowledgeBase:
        normalized_name = name.strip()
        if not normalized_name:
            raise ValueError("Knowledge base name cannot be empty.")
        config = KnowledgeBaseConfig()
        knowledge_base = KnowledgeBase(
            id=str(uuid4()), name=normalized_name, config=config, source_path=source_path
        )
        importer = self._registry.importer(config.source_type)
        documents = importer.load(source_path, knowledge_base_id=knowledge_base.id)
        if not documents:
            raise ValueError("No supported files were found at the specified path.")
        self._registry.backend(config.retrieval_type).indexer().upsert_documents(documents)
        self._repository.create(knowledge_base)
        return knowledge_base

    def retrieve(self, knowledge_base: KnowledgeBase, question: str) -> list[Evidence]:
        return self._registry.backend(knowledge_base.config.retrieval_type).retriever().retrieve(
            RetrievalQuery(
                text=question,
                mode="keyword",
                top_k=5,
                filters={"knowledge_base_id": knowledge_base.id},
            )
        )
