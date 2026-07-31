"""Pure UI state transitions; side effects are requested as actions."""

from __future__ import annotations

from dataclasses import dataclass, replace

from .events import InputChanged, KeyPressed
from .config import Page
from .state import UiState


@dataclass(frozen=True)
class UiAction:
    kind: str
    value: str | None = None


@dataclass(frozen=True)
class Transition:
    state: UiState
    action: UiAction | None = None


def update(
    state: UiState, event: InputChanged | KeyPressed, *, page: Page,
    home_page: str, palette_page: str, item_count: int = 1,
) -> Transition:
    """Return the next state without calling an adapter or terminal API."""
    if page.interaction == "progress":
        return Transition(state)

    # ── text pages with a back route ──────────────────────────────────
    if page.interaction == "text" and page.back_route is not None:
        if isinstance(event, InputChanged):
            # Typing auto-switches to input mode.
            return Transition(replace(state, input_text=event.text, selected_index=1))
        if event.key == "escape":
            return Transition(replace(state, screen=page.back_route, selected_index=0, input_text=""))
        if event.key in {"up", "down"}:
            delta = -1 if event.key == "up" else 1
            return Transition(replace(state, selected_index=(state.selected_index + delta) % 2))
        if event.key == "enter":
            if state.selected_index == 0:
                # User selected the back item.
                return Transition(replace(state, screen=page.back_route, selected_index=0, input_text=""))
            if page.submit_action is None:
                return Transition(state)
            return Transition(state, UiAction(page.submit_action, state.input_text))
        return Transition(state)

    if isinstance(event, InputChanged):
        if state.screen == home_page and event.text == "/":
            return Transition(replace(state, screen=palette_page, input_text="/", selected_index=0))
        if state.screen == palette_page and event.text == "":
            return Transition(replace(state, screen=home_page, input_text="", selected_index=0))
        return Transition(replace(state, input_text=event.text))

    if event.key == "escape" and state.screen != home_page:
        return Transition(replace(state, screen=home_page, input_text="", selected_index=0))
    if event.key in {"up", "down"}:
        delta = -1 if event.key == "up" else 1
        return Transition(replace(state, selected_index=(state.selected_index + delta) % max(item_count, 1)))
    if event.key != "enter":
        return Transition(state)

    if page.submit_action is None:
        return Transition(state)
    if page.interaction == "chat" and not state.input_text.strip():
        return Transition(state)
    value = str(state.selected_index) if page.interaction == "menu" else state.input_text
    if page.interaction == "chat":
        value = state.input_text.strip()
    return Transition(state, UiAction(page.submit_action, value))
