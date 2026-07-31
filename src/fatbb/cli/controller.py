"""Coordinates pure UI transitions with application use cases."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from importlib import import_module
import threading
from typing import cast

from fatbb.application.knowledge_base_service import KnowledgeBaseService
from fatbb.domain.knowledge_base import KnowledgeBase

from .config import CliConfig
from .events import InputChanged, Key, KeyPressed
from .state import UiState
from .update import Transition, update


class CliController:
    """Translate CLI events into UI transitions and knowledge-base operations."""

    def __init__(self, service: KnowledgeBaseService, config: CliConfig):
        """Initialize the controller with the application service it orchestrates."""
        self._service = service
        self._config = config
        self.state = UiState(screen=config.home_page)
        self._app: object = None
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._indexing_lock = threading.Lock()
        self._existing: list[KnowledgeBase] = []
        self._active_knowledge_base: KnowledgeBase | None = None

    def report_progress(self, description: str, current: int, total: int) -> None:
        """Thread-safe: update the progress line shown in the TUI body.

        Called from a background thread during indexing.  Under CPython's GIL
        both ``self.state =`` (a single pointer-store) and
        ``app.invalidate()`` (a flag write) are safe to perform from any
        thread, so we avoid the ``call_from_executor`` round-trip that can
        stall on Windows event loops.
        """
        self.state = replace(
            self.state,
            progress=f"{description}: {current}/{total}" if total else description,
        )
        app = self._app
        if app is not None:
            app.invalidate()  # type: ignore[union-attr]

    def submit_background_indexing(self, task: Callable[[], None]) -> bool:
        """Run *task* on the shared executor, skipping if indexing is already active.

        Returns ``True`` when the task was submitted, ``False`` when an
        indexing operation was already in progress.
        """
        if not self._indexing_lock.acquire(blocking=False):
            return False
        def _wrapper() -> None:
            try:
                task()
            finally:
                self._indexing_lock.release()
        self._executor.submit(_wrapper)
        return True

    def items(self) -> tuple[str, ...]:
        """Resolve the active page's configured item source."""
        choices = self._config.menu_items(self.state.screen)
        if choices is not None:
            return tuple(cast(str, choice.label) for choice in choices)
        source = self._config.item_source(self.state.screen)
        if source is None:
            return ()
        module_name, function_name = source.handler.split(":", maxsplit=1)
        handler = getattr(import_module(module_name), function_name)
        if not callable(handler):
            raise TypeError(f"Configured item source is not callable: {source.handler}")
        typed_handler = cast(Callable[[CliController], tuple[str, ...]], handler)
        return typed_handler(self)

    def _existing_knowledge_base_items(self) -> tuple[str, ...]:
        """Expose the currently loaded local catalog as selectable labels."""
        return tuple(kb.name for kb in self._existing) or ("No knowledge bases found",)

    def on_input_changed(self, text: str) -> None:
        """Update the state machine after the user changes the text input."""
        self._apply(update(
            self.state, InputChanged(text), page=self._config.page(self.state.screen),
            home_page=self._config.home_page, palette_page=self._config.palette_page,
            item_count=len(self.items()),
        ))

    def on_key_pressed(self, key: Key) -> None:
        """Handle a keyboard event, including the terminal's Ctrl-D exit signal."""
        if key == "ctrl_d":
            raise EOFError
        self._apply(update(
            self.state, KeyPressed(key), page=self._config.page(self.state.screen),
            home_page=self._config.home_page, palette_page=self._config.palette_page,
            item_count=len(self.items()),
        ))

    def is_home_page(self) -> bool:
        """Whether the active page is the configured chat/home page."""
        return self.state.screen == self._config.home_page

    def is_palette_page(self) -> bool:
        """Whether the active page is the configured command palette."""
        return self.state.screen == self._config.palette_page

    def is_progress_page(self) -> bool:
        """Whether a background knowledge-base creation task owns the UI."""
        return self._config.page(self.state.screen).interaction == "progress"

    def page_hint(self) -> str | None:
        """Return the active page's optional presentation hint."""
        return self._config.page(self.state.screen).hint

    def _apply(self, transition: Transition) -> None:
        """Commit a UI transition, then perform its optional application action."""
        self.state = transition.state
        if transition.action is not None:
            self._run(transition.action.kind, transition.action.value)

    def _run(self, kind: str, value: str | None) -> None:
        """Dispatch a state-machine action and display recoverable errors in the UI."""
        try:
            action = self._config.action(kind)
            if action is None:
                raise ValueError(f"Unknown UI action: {kind}")
            if action.menu is not None:
                choices = self._config.menu_items(action.menu) or ()
                try:
                    choice = choices[int(value or "0")]
                except IndexError as error:
                    raise ValueError(f"Menu selection is out of range for page: {action.menu}") from error
                if choice.handler is None:
                    raise ValueError(f"No handler is configured for menu item: {choice.value}")
                self._invoke(choice.handler)
                return
            handler_path = action.handler
            if handler_path is None:
                raise ValueError(f"No handler is configured for UI action: {kind}")
            self._invoke(handler_path, value)
        except (RuntimeError, ValueError) as error:
            self.state = replace(self.state, status=f"Error: {error}")

    def _invoke(self, handler_path: str, value: str | None = None) -> None:
        """Load one configured action handler and invoke it with this controller."""
        try:
            module_name, function_name = handler_path.split(":", maxsplit=1)
            handler = getattr(import_module(module_name), function_name)
            if not callable(handler):
                raise TypeError(f"Configured action handler is not callable: {handler_path}")
            if value is None:
                handler(self)
            else:
                handler(self, value)
        except (ImportError, AttributeError, TypeError, ValueError) as error:
            raise RuntimeError(f"Unable to invoke configured action handler: {handler_path}") from error
