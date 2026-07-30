"""Concrete CLI actions referenced by the source-controlled UI configuration."""

from __future__ import annotations

import re
from dataclasses import replace
from typing import TYPE_CHECKING

from fatbb.domain.knowledge_base import KnowledgeBase

from .state import Screen

if TYPE_CHECKING:
    from .controller import CliController


_DATABASE_URL_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:(?://|\S+)")


def open_existing_knowledge_bases(controller: CliController) -> None:
    """Load the local KB catalog before displaying the selection page."""
    controller._existing = controller._service.list()
    controller.state = replace(
        controller.state, screen=Screen.EXISTING_KNOWLEDGE_BASES, selected_index=0,
    )


def open_knowledge_base_creation(controller: CliController) -> None:
    """Begin the configured knowledge-base creation flow."""
    controller.state = replace(controller.state, screen=Screen.RETRIEVAL_TYPE, selected_index=0)


def return_to_chat(controller: CliController) -> None:
    """Close a menu and return to the chat page."""
    controller.state = replace(controller.state, screen=Screen.CHAT, selected_index=0)


def select_knowledge_base(controller: CliController, value: str | None) -> None:
    """Make the selected existing knowledge base active for chat retrieval."""
    if not controller._existing:
        controller.state = replace(controller.state, status="No knowledge bases are available.")
        return
    _activate(controller, controller._existing[int(value or "0")], "Knowledge base selected.")


def set_knowledge_base_name(controller: CliController, value: str | None) -> None:
    """Validate a new knowledge-base name and advance to source setup."""
    name = value or ""
    if not name.strip():
        raise ValueError("Knowledge base name cannot be empty.")
    controller.state = replace(
        controller.state, screen=Screen.SOURCE_PATH, input_text="", pending_name=name.strip(),
        status="Enter a local file or directory path.",
    )


def set_database_url(controller: CliController, value: str | None) -> None:
    """Store a database URL while building a new knowledge base."""
    database_url = (value or "").strip()
    if not database_url:
        raise ValueError("Database URL cannot be empty.")
    # Terminal pastes on Windows can introduce CRLF line endings, and
    # accidental multi-line clipboard content can embed the database name
    # before the URL. Extract the intended single-line connection string.
    if "\n" in database_url or "\r" in database_url:
        lines = [
            line.strip()
            for line in database_url.replace("\r", "\n").split("\n")
        ]
        # Prefer a line that looks like a database connection URI. This
        # supports registered database backends beyond PostgreSQL.
        url_lines = [
            line for line in lines
            if _DATABASE_URL_PATTERN.match(line)
        ]
        if url_lines:
            database_url = url_lines[-1]
            controller.state = replace(
                controller.state, screen=Screen.SOURCE_TYPE, input_text="",
                pending_database_url=database_url, selected_index=0,
                status="URL extracted from multi-line paste.",
            )
            return
        # Fall back to collapsing all whitespace when no URI line is found.
        collapsed = " ".join(lines)
        if not collapsed:
            raise ValueError("Database URL cannot be empty.")
        database_url = collapsed
        controller.state = replace(
            controller.state, screen=Screen.SOURCE_TYPE, input_text="",
            pending_database_url=database_url, selected_index=0,
            status="URL whitespace was collapsed.",
        )
        return
    controller.state = replace(
        controller.state, screen=Screen.SOURCE_TYPE, input_text="",
        pending_database_url=database_url, selected_index=0,
    )


def set_retrieval_type(controller: CliController, value: str | None) -> None:
    choices = controller._config.menu_items(Screen.RETRIEVAL_TYPE.value) or ()
    controller.state = replace(
        controller.state, pending_retrieval_type=choices[int(value or "0")].value,
        screen=Screen.DATABASE_TYPE, selected_index=0,
    )


def set_database_type(controller: CliController, value: str | None) -> None:
    choices = controller._config.menu_items(Screen.DATABASE_TYPE.value) or ()
    controller.state = replace(
        controller.state, pending_database_type=choices[int(value or "0")].value,
        screen=Screen.DATABASE_URL, input_text="", selected_index=0,
    )


def set_source_type(controller: CliController, value: str | None) -> None:
    choices = controller._config.menu_items(Screen.SOURCE_TYPE.value) or ()
    controller.state = replace(
        controller.state, pending_source_type=choices[int(value or "0")].value,
        screen=Screen.KNOWLEDGE_BASE_NAME, input_text="", selected_index=0,
    )


def create_knowledge_base(controller: CliController, value: str | None) -> None:
    """Create and index a knowledge base from the pending configuration."""
    state = controller.state
    knowledge_base = controller._service.create(
        state.pending_name, state.pending_retrieval_type, state.pending_database_type,
        state.pending_database_url, state.pending_source_type, (value or "").strip(),
    )
    _activate(controller, knowledge_base, f'Indexed and selected "{knowledge_base.name}".')


def retrieve(controller: CliController, value: str | None) -> None:
    """Retrieve evidence for a question using the active knowledge base."""
    if not controller.state.active_knowledge_base_id:
        controller.state = replace(
            controller.state, input_text="", status="Select a knowledge base with / first.",
        )
        return
    knowledge_base = controller._active_knowledge_base
    if knowledge_base is None:
        raise RuntimeError("The active knowledge base could not be loaded.")
    evidence = controller._service.retrieve(knowledge_base, value or "")
    lines = tuple(_format_evidence(item, index + 1) for index, item in enumerate(evidence))
    controller.state = replace(
        controller.state, input_text="", lines=lines,
        status=(f"Retrieved {len(evidence)} relevant sources." if evidence else "No relevant sources found."),
    )


def _activate(controller: CliController, knowledge_base: KnowledgeBase, status: str) -> None:
    """Switch to chat with a knowledge base and clear transient UI content."""
    controller._active_knowledge_base = knowledge_base
    controller.state = replace(
        controller.state, screen=Screen.CHAT, input_text="", selected_index=0,
        active_knowledge_base_id=knowledge_base.id,
        active_knowledge_base_name=knowledge_base.name, status=status, lines=(),
    )


def _format_evidence(item: object, index: int) -> str:
    """Convert an evidence object into a compact, readable CLI result entry."""
    from rag.models.evidence import Evidence

    evidence = item if isinstance(item, Evidence) else None
    if evidence is None:
        return str(item)
    source = evidence.source.title if evidence.source and evidence.source.title else "Unknown source"
    content = " ".join(evidence.content.split())
    preview = content[:300] + ("…" if len(content) > 300 else "")
    return f"{index}. {source} · score {evidence.score:.2f}\n   {preview}"
