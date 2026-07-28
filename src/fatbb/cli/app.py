"""Executable FatBB terminal application."""

from __future__ import annotations

import os

from prompt_toolkit.application import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, Layout
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.containers import Window
from prompt_toolkit.layout.dimension import Dimension

from fatbb.application.knowledge_base_service import KnowledgeBaseService
from fatbb.application.registry import CapabilityRegistry
from fatbb.infrastructure.bm25 import PostgresBm25Backend
from fatbb.infrastructure.local_files import LocalFileImporter
from fatbb.infrastructure.postgres_knowledge_bases import PostgresKnowledgeBaseRepository

from .controller import CliController
from .view import body, header, prompt


def build_controller(database_url: str) -> CliController:
    registry = CapabilityRegistry()
    registry.register_backend(PostgresBm25Backend(database_url))
    registry.register_importer(LocalFileImporter())
    service = KnowledgeBaseService(PostgresKnowledgeBaseRepository(database_url), registry)
    return CliController(service)


def main() -> None:
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise SystemExit("DATABASE_URL is required. See README.md for setup instructions.")
    controller = build_controller(database_url)
    app: Application[None]
    syncing = False

    def redraw() -> None:
        app.invalidate()

    def set_buffer(text: str) -> None:
        nonlocal syncing
        syncing = True
        buffer.text = text
        buffer.cursor_position = len(text)
        syncing = False

    def changed(buffer: Buffer) -> None:
        if syncing:
            return
        controller.on_input_changed(buffer.text)
        redraw()

    buffer = Buffer(on_text_changed=changed)
    key_bindings = KeyBindings()

    @key_bindings.add("up")
    def _up(event) -> None:
        controller.on_key_pressed("up")
        redraw()

    @key_bindings.add("down")
    def _down(event) -> None:
        controller.on_key_pressed("down")
        redraw()

    @key_bindings.add("enter")
    def _enter(event) -> None:
        controller.on_key_pressed("enter")
        set_buffer(controller.state.input_text)
        redraw()

    @key_bindings.add("escape")
    def _escape(event) -> None:
        controller.on_key_pressed("escape")
        set_buffer("")
        redraw()

    @key_bindings.add("c-d")
    def _exit(event) -> None:
        event.app.exit()

    layout = Layout(
        HSplit(
            [
                Window(FormattedTextControl(lambda: header(controller)), height=1),
                Window(FormattedTextControl(lambda: body(controller)), wrap_lines=True),
                Window(height=1, char="─"),
                Window(
                    BufferControl(buffer=buffer, focusable=True),
                    height=Dimension.exact(1),
                    get_line_prefix=lambda _line, _wrap: [("class:prompt", prompt(controller))],
                ),
            ]
        )
    )
    app = Application(layout=layout, key_bindings=key_bindings, full_screen=True, mouse_support=False)
    app.run()


if __name__ == "__main__":
    main()
