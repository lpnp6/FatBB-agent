"""Knowledge-base configuration independent of a concrete adapter."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KnowledgeBaseConfig:
    """Immutable capability choices used to build a knowledge-base runtime.

    Values are stable identifiers from ``CapabilityRegistry`` rather than
    implementation classes. This lets a stored knowledge base keep working
    when its adapter implementation changes, and allows new capabilities to be
    registered without changing the CLI state machine.
    """

    # The retrieval adapter key selected from kb.toml.
    retrieval_type: str
    # The persistence/runtime provider key selected from kb.toml.
    database_type: str
    # Connection string supplied while creating the knowledge base. It is
    # persisted so the application has no global DATABASE_URL requirement.
    database_url: str
    # The ingestion adapter key. ``file_path`` imports local text files.
    source_type: str
    # Embedding provider configuration is required by vector retrieval only.
    embedding_provider: str | None = None
    embedding_model: str | None = None
    embedding_url: str | None = None


@dataclass(frozen=True)
class KnowledgeBase:
    """A named, isolated collection of indexed source documents.

    ``id`` is the opaque storage identity used as the metadata filter during
    retrieval. ``name`` is only the user-facing label and may therefore never
    be used to scope queries. ``source_path`` records the initial import
    location so a future reindex workflow can reuse it. The database URL is
    stored in ``config`` because each knowledge base can use a different DB.
    """

    # Opaque identifier persisted in ``knowledge_bases`` and chunk metadata.
    id: str
    # Human-readable, unique label selected in the terminal UI.
    name: str
    # Backend and ingestion choices captured at creation time.
    config: KnowledgeBaseConfig
    # Original user-provided local path for the current file-path importer.
    source_path: str
