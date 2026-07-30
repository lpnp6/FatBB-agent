"""Pure UI state transitions; side effects are requested as actions."""

from __future__ import annotations

from dataclasses import dataclass, replace

from .events import InputChanged, KeyPressed
from .state import Screen, UiState


@dataclass(frozen=True)
class UiAction:
    kind: str
    value: str | None = None


@dataclass(frozen=True)
class Transition:
    state: UiState
    action: UiAction | None = None


def update(state: UiState, event: InputChanged | KeyPressed, *, item_count: int = 1) -> Transition:
    """Return the next state without calling an adapter or terminal API."""
    if isinstance(event, InputChanged):
        if state.screen is Screen.CHAT and event.text == "/":
            return Transition(replace(state, screen=Screen.PALETTE, input_text="/", selected_index=0))
        if state.screen is Screen.PALETTE and event.text == "":
            return Transition(replace(state, screen=Screen.CHAT, input_text="", selected_index=0))
        return Transition(replace(state, input_text=event.text))

    if event.key == "escape" and state.screen is not Screen.CHAT:
        return Transition(replace(state, screen=Screen.CHAT, input_text="", selected_index=0))
    if event.key in {"up", "down"}:
        delta = -1 if event.key == "up" else 1
        return Transition(replace(state, selected_index=(state.selected_index + delta) % max(item_count, 1)))
    if event.key != "enter":
        return Transition(state)

    if state.screen is Screen.PALETTE:
        if item_count > 1 and state.selected_index == item_count - 1:
            return Transition(replace(state, screen=Screen.CHAT, input_text="", selected_index=0))
        return Transition(replace(state, screen=Screen.KNOWLEDGE_BASE_MENU, input_text=""))
    if state.screen is Screen.KNOWLEDGE_BASE_MENU:
        return Transition(
            replace(state, input_text="", selected_index=0),
            UiAction("knowledge_base_menu_selection", str(state.selected_index)),
        )
    if state.screen is Screen.EXISTING_KNOWLEDGE_BASES:
        return Transition(state, UiAction("select_knowledge_base", str(state.selected_index)))
    if state.screen is Screen.RETRIEVAL_TYPE:
        return Transition(state, UiAction("set_retrieval_type", str(state.selected_index)))
    if state.screen is Screen.DATABASE_TYPE:
        return Transition(state, UiAction("set_database_type", str(state.selected_index)))
    if state.screen is Screen.DATABASE_URL:
        return Transition(state, UiAction("set_database_url", state.input_text))
    if state.screen is Screen.SOURCE_TYPE:
        return Transition(state, UiAction("set_source_type", str(state.selected_index)))
    if state.screen is Screen.KNOWLEDGE_BASE_NAME:
        return Transition(state, UiAction("set_knowledge_base_name", state.input_text))
    if state.screen is Screen.SOURCE_PATH:
        return Transition(state, UiAction("create_knowledge_base", state.input_text))
    if state.screen is Screen.CHAT and state.input_text.strip():
        return Transition(state, UiAction("retrieve", state.input_text.strip()))
    return Transition(state)
