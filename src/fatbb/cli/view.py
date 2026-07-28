"""Prompt-toolkit rendering kept separate from UI state transitions."""

from __future__ import annotations

from prompt_toolkit.formatted_text import HTML

from .controller import CliController
from .state import Screen


def header(controller: CliController) -> HTML:
    state = controller.state
    current = state.active_knowledge_base_name or "No knowledge base selected"
    return HTML(f"<b>FatBB</b>  <ansibrightblack>· {current}</ansibrightblack>")


def body(controller: CliController) -> HTML:
    state = controller.state
    parts = [state.status]
    if state.lines:
        parts.extend(("", *state.lines))
    if controller.items():
        parts.extend(("", _menu_text(controller)))
    if state.screen is Screen.KNOWLEDGE_BASE_NAME:
        parts.append("Enter a knowledge base name.")
    return HTML("\n".join(_escape(part) for part in parts))


def prompt(controller: CliController) -> str:
    return "FatBB > "


def _menu_text(controller: CliController) -> str:
    return "\n".join(
        f"{'>' if index == controller.state.selected_index else ' '} {item}"
        for index, item in enumerate(controller.items())
    )


def _escape(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
