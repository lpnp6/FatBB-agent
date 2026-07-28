"""Immutable presentation state. It has no terminal or database dependency."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class Screen(StrEnum):
    CHAT = "chat"
    PALETTE = "palette"
    KNOWLEDGE_BASE_MENU = "knowledge_base_menu"
    EXISTING_KNOWLEDGE_BASES = "existing_knowledge_bases"
    RETRIEVAL_TYPE = "retrieval_type"
    DATABASE_TYPE = "database_type"
    SOURCE_TYPE = "source_type"
    KNOWLEDGE_BASE_NAME = "knowledge_base_name"
    SOURCE_PATH = "source_path"


@dataclass(frozen=True)
class UiState:
    screen: Screen = Screen.CHAT
    input_text: str = ""
    selected_index: int = 0
    active_knowledge_base_id: str | None = None
    active_knowledge_base_name: str | None = None
    pending_name: str = ""
    pending_source_path: str = ""
    status: str = "Type / to open the command palette."
    lines: tuple[str, ...] = field(default_factory=tuple)
