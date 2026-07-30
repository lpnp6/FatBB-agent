"""Tests for knowledge-base creation sequencing."""

from __future__ import annotations

import unittest

from fatbb.application.knowledge_base_service import KnowledgeBaseService
from rag.models import Document, SourceRef


class RecordingRepository:
    def __init__(self) -> None:
        self.created = []

    def list_knowledge_bases(self):
        return self.created

    def create_knowledge_base(self, knowledge_base) -> None:
        self.created.append(knowledge_base)


class RecordingImporter:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def load(self, path: str, *, knowledge_base_id: str):
        self.events.append("load")
        return [
            Document(
                id="document-1",
                content="A source document.",
                source=SourceRef(uri="file:///source.txt"),
                metadata={"knowledge_base_id": knowledge_base_id},
            )
        ]


class RecordingAdapter:
    def __init__(self, events: list[str], fail_connection: bool = False) -> None:
        self.events = events
        self.fail_connection = fail_connection

    def check_connection(self, database_url: str) -> None:
        self.events.append("check_connection")
        if self.fail_connection:
            raise ConnectionError("database is unavailable")

    def indexer(self, database_url: str):
        return self

    def upsert_documents(self, documents) -> None:
        self.events.append("upsert_documents")

    def retriever(self, database_url: str):
        raise AssertionError("retriever is not used during creation")


class RecordingRegistry:
    def __init__(self, adapter: RecordingAdapter, importer: RecordingImporter) -> None:
        self.adapter = adapter
        self._importer = importer

    def knowledge_base(self, retrieval_type: str, database_type: str) -> RecordingAdapter:
        return self.adapter

    def importer(self, source_type: str) -> RecordingImporter:
        return self._importer


class KnowledgeBaseServiceTests(unittest.TestCase):
    def test_checks_database_before_loading_and_indexing_documents(self) -> None:
        events: list[str] = []
        repository = RecordingRepository()
        service = KnowledgeBaseService(
            repository,
            RecordingRegistry(RecordingAdapter(events), RecordingImporter(events)),
        )

        service.create("Docs", "bm25", "pg", "postgresql://db", "file_path", "/source")

        self.assertEqual(events, ["check_connection", "load", "upsert_documents"])
        self.assertEqual(len(repository.created), 1)

    def test_does_not_load_documents_when_connection_check_fails(self) -> None:
        events: list[str] = []
        service = KnowledgeBaseService(
            RecordingRepository(),
            RecordingRegistry(RecordingAdapter(events, fail_connection=True), RecordingImporter(events)),
        )

        with (
            self.assertRaisesRegex(ValueError, "Could not connect to the configured pg database"),
            self.assertLogs("fatbb.application.knowledge_base_service", level="ERROR"),
        ):
            service.create("Docs", "bm25", "pg", "postgresql://db", "file_path", "/source")

        self.assertEqual(events, ["check_connection"])
