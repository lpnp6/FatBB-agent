"""Small explicit registry for future retrieval and source implementations."""

from __future__ import annotations

from fatbb.domain.ports import RetrievalBackend, SourceImporter


class CapabilityRegistry:
    def __init__(self) -> None:
        self._backends: dict[str, RetrievalBackend] = {}
        self._importers: dict[str, SourceImporter] = {}

    def register_backend(self, backend: RetrievalBackend) -> None:
        self._backends[backend.type] = backend

    def register_importer(self, importer: SourceImporter) -> None:
        self._importers[importer.type] = importer

    def backend(self, retrieval_type: str) -> RetrievalBackend:
        try:
            return self._backends[retrieval_type]
        except KeyError as error:
            raise ValueError(f"Unsupported retrieval type: {retrieval_type}") from error

    def importer(self, source_type: str) -> SourceImporter:
        try:
            return self._importers[source_type]
        except KeyError as error:
            raise ValueError(f"Unsupported source type: {source_type}") from error
