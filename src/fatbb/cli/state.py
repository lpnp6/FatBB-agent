"""Immutable presentation state. It has no terminal or database dependency."""

from __future__ import annotations

from dataclasses import dataclass, field
@dataclass(frozen=True)
class UiState:
    # Page identifiers are declared in ``config/cli.toml``.  The controller
    # supplies the configured initial page when it creates this state.
    screen: str = ""
    input_text: str = ""
    selected_index: int = 0
    active_knowledge_base_id: str | None = None
    active_knowledge_base_name: str | None = None
    pending_name: str = ""
    pending_retrieval_type: str = ""
    pending_database_type: str = ""
    pending_database_url: str = ""
    pending_embedding_provider: str = ""
    pending_embedding_model: str = ""
    pending_embedding_url: str = ""
    pending_source_type: str = ""
    pending_source_path: str = ""
    status: str = "Type / to open the command palette."
    progress: str = ""
    lines: tuple[str, ...] = field(default_factory=tuple)
