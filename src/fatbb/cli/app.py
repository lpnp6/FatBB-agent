"""Executable FatBB terminal application."""

from __future__ import annotations

import logging
from pathlib import Path

from prompt_toolkit.application import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.filters import Condition
from prompt_toolkit.layout import Float, FloatContainer, HSplit, Layout
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.containers import ConditionalContainer, Window
from prompt_toolkit.layout.dimension import Dimension

from fatbb.application.knowledge_base_service import KnowledgeBaseService
from fatbb.application.registry import CapabilityRegistry
from fatbb.infrastructure.local.local import Local

from .controller import CliController
from .config import CliConfig
from .view import body, header, palette, prompt


def _configure_logging() -> None:
    """Write operational logs to a local file without disturbing the terminal UI."""
    logger = logging.getLogger()
    if any(isinstance(handler, logging.FileHandler) for handler in logger.handlers):
        return
    log_directory = Path.home() / ".fatbb"
    log_directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    handler = logging.FileHandler(log_directory / "fatbb.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s"
    ))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def build_controller() -> CliController:
    """Compose the CLI from configuration-driven capabilities and local state.

    This is the application's composition root: it selects no concrete
    retrieval or ingestion adapter itself. ``CapabilityRegistry`` reads the
    source-controlled catalog and constructs the adapter requested later by a
    selected knowledge base's saved type identifiers.
    """
    # ``fatbb/config`` is packaged alongside this module, so the same catalog
    # is available when running from source or through the installed command.
    config_directory = Path(__file__).resolve().parents[1] / "config"
    # Registry owns KB module imports; CLI configuration never names adapters.
    registry = CapabilityRegistry(config_directory / "kb.toml")
    # The CLI catalog is deliberately local; PostgreSQL is only a knowledge
    # base's retrieval/index storage, never the store for CLI configuration.
    # ``local`` owns all user-machine state: knowledge-base configuration now,
    # and conversation data or other local concerns in later iterations.
    local = Local()
    service = KnowledgeBaseService(local, registry)

    # CLI presentation is separate from KB configuration and registry imports.
    config = CliConfig(config_directory / "cli.toml")
    return CliController(service, config)


def main() -> None:
    """Create the prompt-toolkit runtime and bridge terminal events to the controller."""
    _configure_logging()
    controller = build_controller()
    app: Application[None]
    syncing = False

    def redraw() -> None:
        """Request a render after a state-changing controller operation."""
        app.invalidate()

    def set_buffer(text: str) -> None:
        """Synchronize programmatic state changes without re-emitting input events."""
        nonlocal syncing
        syncing = True
        buffer.text = text
        buffer.cursor_position = len(text)
        syncing = False

    def changed(buffer: Buffer) -> None:
        """Forward typed text; the controller decides whether `/` opens a menu."""
        if syncing:
            return
        controller.on_input_changed(buffer.text)
        redraw()

    buffer = Buffer(on_text_changed=changed)
    key_bindings = KeyBindings()

    @key_bindings.add("up")
    def _up(event) -> None:
        # Selection movement is meaningful only on screens with menu items;
        # the controller's state reducer safely handles every screen.
        controller.on_key_pressed("up")
        redraw()

    @key_bindings.add("down")
    def _down(event) -> None:
        controller.on_key_pressed("down")
        redraw()

    @key_bindings.add("enter")
    def _enter(event) -> None:
        # Enter may select a menu item, advance the creation flow, or submit a
        # chat query. The controller returns the canonical next input text.
        controller.on_key_pressed("enter")
        set_buffer(controller.state.input_text)
        redraw()

    @key_bindings.add("escape")
    def _escape(event) -> None:
        # Close transient menus and clear their triggering input.
        controller.on_key_pressed("escape")
        set_buffer("")
        redraw()

    @key_bindings.add("c-d")
    def _exit(event) -> None:
        if controller.is_progress_page():
            return
        event.app.exit()

    @key_bindings.add("c-c")
    def _interrupt(event) -> None:
        if controller.is_progress_page():
            return
        event.app.exit()

    @key_bindings.add("pageup")
    def _pageup(event) -> None:
        controller.on_scroll("page_up")
        redraw()

    @key_bindings.add("pagedown")
    def _pagedown(event) -> None:
        controller.on_scroll("page_down")
        redraw()

    @key_bindings.add("c-up")
    def _ctrl_up(event) -> None:
        controller.on_scroll("wheel_up")
        redraw()

    @key_bindings.add("c-down")
    def _ctrl_down(event) -> None:
        controller.on_scroll("wheel_down")
        redraw()

    @key_bindings.add("<scroll-up>")
    def _wheel_up(event) -> None:
        controller.on_scroll("wheel_up")
        redraw()

    @key_bindings.add("<scroll-down>")
    def _wheel_down(event) -> None:
        controller.on_scroll("wheel_down")
        redraw()

    # The chat surface always remains mounted. The slash-command palette is a
    # floating overlay that covers the entire body, while the header and input
    # remain visible and the previous transcript stays preserved in state.
    chat = HSplit(
        [
            Window(FormattedTextControl(lambda: header(controller)), height=1),
            Window(FormattedTextControl(lambda: body(controller)), wrap_lines=True),
            ConditionalContainer(
                HSplit([
                    Window(height=1, char="─"),
                    Window(
                        BufferControl(buffer=buffer, focusable=True),
                        height=Dimension.exact(1),
                        get_line_prefix=lambda _line, _wrap: [("class:prompt", prompt(controller))],
                    ),
                ]),
                filter=Condition(lambda: not controller.is_progress_page()),
            ),
        ]
    )
    command_palette = Window(
        FormattedTextControl(lambda: palette(controller)),
        wrap_lines=True,
    )
    layout = Layout(
        FloatContainer(
            content=chat,
            floats=[
                Float(
                    content=ConditionalContainer(
                        command_palette,
                        filter=Condition(controller.is_palette_page),
                    ),
                    # Header is one row; separator + prompt consume two rows
                    # at the bottom. Fill everything between them.
                    top=1,
                    bottom=2,
                    left=0,
                    right=0,
                    hide_when_covering_content=False,
                    transparent=False,
                    z_index=1,
                )
            ],
        )
    )
    # ``full_screen`` permits a stable overlay layout; the controller remains
    # terminal-framework independent and can be reused by another UI later.
    app = Application(layout=layout, key_bindings=key_bindings, full_screen=True, mouse_support=True)
    controller._app = app
    import shutil
    controller.terminal_height = shutil.get_terminal_size().lines
    app.run()


if __name__ == "__main__":
    main()
