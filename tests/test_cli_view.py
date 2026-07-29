"""Rendering tests for the terminal chat and command-palette views."""

from types import SimpleNamespace
import unittest

from prompt_toolkit.formatted_text import to_formatted_text

from fatbb.cli.state import Screen, UiState
from fatbb.cli.view import body, palette


class CliViewTests(unittest.TestCase):
    def test_palette_is_rendered_separately_from_chat_body(self) -> None:
        controller = SimpleNamespace(
            state=UiState(screen=Screen.PALETTE, input_text="/", lines=("Previous result",)),
            items=lambda: ("Knowledge Base",),
        )

        chat_text = _plain_text(body(controller))
        palette_text = _plain_text(palette(controller))

        self.assertIn("Previous result", chat_text)
        self.assertNotIn("Knowledge Base", chat_text)
        self.assertIn("Command palette", palette_text)
        self.assertIn("> Knowledge Base", palette_text)


def _plain_text(value: object) -> str:
    return "".join(text for _style, text, *_ in to_formatted_text(value))

