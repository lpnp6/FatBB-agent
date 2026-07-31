"""Prompt-toolkit rendering kept separate from UI state transitions."""

from __future__ import annotations

from prompt_toolkit.formatted_text import HTML

from .controller import CliController


def header(controller: CliController) -> HTML:
    state = controller.state
    current = state.active_knowledge_base_name or "No knowledge base selected"
    return HTML(f"<b>FatBB</b>  <ansibrightblack>· {current}</ansibrightblack>")


def body(controller: CliController) -> HTML:
    state = controller.state
    # Menus and creation pages are modal over the chat transcript. Keep prior
    # retrieval results in state, but do not append a new menu beneath them.
    # Returning to CHAT makes the unchanged transcript visible again.
    parts: list[str] = []
    if controller.is_progress_page():
        parts.extend(("Creating knowledge base", state.progress or "Starting…"))
    elif controller.is_home_page():
        parts.append(state.status)
        if state.progress:
            parts.append(f"  {state.progress}")
        if state.lines:
            parts.extend(("", *state.lines))
    elif state.status.startswith("Error:"):
        parts.append(state.status)
    if hint := controller.page_hint():
        parts.append(hint)
    if controller.items() and not controller.is_palette_page():
        parts.extend(("", _menu_text(controller)))
    return HTML("\n".join(_escape(part) for part in parts))


def palette(controller: CliController) -> HTML:
    """Render the command palette that floats above the chat view."""
    return HTML("\n".join(_escape(part) for part in ("Command palette", "", _menu_text(controller))))


def prompt(controller: CliController) -> str:
    return "" if controller.is_progress_page() else "FatBB > "


def _menu_text(controller: CliController) -> str:
    return "\n".join(
        f"{'>' if index == controller.state.selected_index else ' '} {item}"
        for index, item in enumerate(controller.items())
    )


def _escape(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
