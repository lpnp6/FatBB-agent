"""Rendering tests for the terminal chat and command-palette views."""

from types import SimpleNamespace
import unittest

from prompt_toolkit.formatted_text import to_formatted_text

from fatbb.cli.state import UiState
from fatbb.cli.view import body, palette


class CliViewTests(unittest.TestCase):
    def test_palette_hides_previous_chat_result_from_modal_body(self) -> None:
        controller = SimpleNamespace(
            state=UiState(screen="palette", input_text="/", lines=("Previous result",)),
            items=lambda: ("Knowledge Base",),
            is_home_page=lambda: False,
            is_palette_page=lambda: True,
            is_progress_page=lambda: False,
            page_hint=lambda: None,
        )

        chat_text = _plain_text(body(controller))
        palette_text = _plain_text(palette(controller))

        self.assertNotIn("Previous result", chat_text)
        self.assertNotIn("Knowledge Base", chat_text)
        self.assertIn("Command palette", palette_text)
        self.assertIn("> Knowledge Base", palette_text)

    def test_progress_page_shows_progress(self) -> None:
        controller = SimpleNamespace(
            state=UiState(screen="indexing", progress="Chunking documents: 1/2"),
            items=lambda: (),
            is_home_page=lambda: False,
            is_palette_page=lambda: False,
            is_progress_page=lambda: True,
            page_hint=lambda: "Please wait.",
        )

        page_text = _plain_text(body(controller))

        self.assertIn("Creating knowledge base", page_text)
        self.assertIn("Chunking documents: 1/2", page_text)


def _plain_text(value: object) -> str:
    return "".join(text for _style, text, *_ in to_formatted_text(value))
