"""Concrete CLI actions referenced by the source-controlled UI configuration."""

from __future__ import annotations

import re
from dataclasses import replace
from typing import TYPE_CHECKING

from fatbb.domain.knowledge_base import KnowledgeBase

if TYPE_CHECKING:
    from .controller import CliController


_DATABASE_URL_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:(?://|\S+)")


def open_existing_knowledge_bases(controller: CliController) -> None:
    """Load the local KB catalog before displaying the selection page."""
    controller._existing = controller._service.list()
    _show(controller, "open_existing_knowledge_bases")


def open_knowledge_base_creation(controller: CliController) -> None:
    """Begin the configured knowledge-base creation flow."""
    _show(controller, "open_knowledge_base_creation")


def open_knowledge_base_menu(controller: CliController) -> None:
    """Open the configured knowledge-base menu from the command palette."""
    _show(controller, "open_knowledge_base_menu")


def return_to_chat(controller: CliController) -> None:
    """Close a menu and return to the chat page."""
    controller.state = replace(controller.state, screen=controller._config.home_page, selected_index=0)


def select_knowledge_base(controller: CliController, value: str | None) -> None:
    """Make the selected existing knowledge base active for chat retrieval."""
    selected_index = int(value or "0")
    back_index = len(controller._existing) if controller._existing else 1
    if selected_index == back_index:
        _show(controller, "existing_knowledge_bases_back")
        return
    if not controller._existing:
        controller.state = replace(controller.state, status="No knowledge bases are available.")
        return
    _activate(controller, controller._existing[int(value or "0")], "Knowledge base selected.")


def set_knowledge_base_name(controller: CliController, value: str | None) -> None:
    """Validate a new knowledge-base name and advance to source setup."""
    name = value or ""
    if not name.strip():
        raise ValueError("Knowledge base name cannot be empty.")
    _show(controller, "knowledge_base_name_next", pending_name=name.strip(),
          status="Enter a local file or directory path.")


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
            _show(controller, "database_url_next", pending_database_url=database_url,
                  status="URL extracted from multi-line paste.")
            return
        # Fall back to collapsing all whitespace when no URI line is found.
        collapsed = " ".join(lines)
        if not collapsed:
            raise ValueError("Database URL cannot be empty.")
        database_url = collapsed
        _show(controller, "database_url_next", pending_database_url=database_url,
              status="URL whitespace was collapsed.")
        return
    _show(controller, "database_url_next", pending_database_url=database_url)


def set_retrieval_type(controller: CliController, value: str | None) -> None:
    choices = controller._config.menu_items(controller._config.route("retrieval_type")) or ()
    if choices[int(value or "0")].value == "back":
        _show(controller, "retrieval_type_back")
        return
    _show(controller, "retrieval_type_next", pending_retrieval_type=choices[int(value or "0")].value)


def set_database_type(controller: CliController, value: str | None) -> None:
    choices = controller._config.menu_items(controller._config.route("database_type")) or ()
    if choices[int(value or "0")].value == "back":
        _show(controller, "database_type_back")
        return
    _show(controller, "database_type_next", pending_database_type=choices[int(value or "0")].value)


def set_source_type(controller: CliController, value: str | None) -> None:
    choices = controller._config.menu_items(controller._config.route("source_type")) or ()
    if choices[int(value or "0")].value == "back":
        _show(controller, "source_type_back")
        return
    _show(controller, "source_type_next", pending_source_type=choices[int(value or "0")].value)


def create_knowledge_base(controller: CliController, value: str | None) -> None:
    """Create and index a knowledge base from the pending configuration.

    The heavy import and indexing work runs in a background thread so the
    terminal UI stays responsive.  Progress is streamed into the body via
    :meth:`CliController.report_progress` and the final result is posted
    back to the chat screen when the work completes.

    Only one indexing operation is allowed at a time; a second attempt
    while indexing is in progress is silently ignored.
    """
    state = controller.state
    app = controller._app
    source_path = (value or "").strip()

    # Snapshot pending configuration before handing it to the worker thread.
    pending_name = state.pending_name
    pending_retrieval_type = state.pending_retrieval_type
    pending_database_type = state.pending_database_type
    pending_database_url = state.pending_database_url
    pending_source_type = state.pending_source_type

    # Show the user that work has started immediately.
    controller.state = replace(
        controller.state,
        status="Indexing documents…",
        progress="Starting…",
    )
    if app is not None:
        app.invalidate()  # type: ignore[union-attr]

    def _run() -> None:
        try:
            kb = controller._service.create(
                pending_name, pending_retrieval_type, pending_database_type,
                pending_database_url, pending_source_type, source_path,
                on_progress=controller.report_progress,
            )
            _activate(controller, kb, f'Indexed and selected "{kb.name}".')
            controller.state = replace(controller.state, progress="")
            if app is not None:
                app.invalidate()  # type: ignore[union-attr]
        except Exception as exc:
            controller.state = replace(
                controller.state,
                status=f"Error: {exc}",
                progress="",
            )
            if app is not None:
                app.invalidate()  # type: ignore[union-attr]

    if not controller.submit_background_indexing(_run):
        controller.state = replace(
            controller.state,
            status="An indexing operation is already in progress.",
            progress="",
        )


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
        controller.state, screen=controller._config.home_page, input_text="", selected_index=0,
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


def _show(controller: CliController, route: str, **changes: object) -> None:
    """Navigate through a TOML-defined route and reset transient input."""
    controller.state = replace(
        controller.state,
        screen=controller._config.route(route),
        input_text="",
        selected_index=0,
        **changes,
    )
