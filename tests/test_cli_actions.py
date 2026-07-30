"""Tests for database URL handling in CLI actions."""

from types import SimpleNamespace
import unittest

from fatbb.cli.actions import select_knowledge_base, set_database_url, set_database_type
from fatbb.cli.state import Screen, UiState


class SetDatabaseUrlTests(unittest.TestCase):
    def test_extracts_a_non_postgresql_database_url_from_a_multiline_paste(self) -> None:
        controller = SimpleNamespace(state=UiState(screen=Screen.DATABASE_URL))

        set_database_url(controller, "database name\nmongodb://localhost:27017/fatbb\n")

        self.assertEqual(
            controller.state.pending_database_url,
            "mongodb://localhost:27017/fatbb",
        )
        self.assertEqual(controller.state.status, "URL extracted from multi-line paste.")

    def test_rejects_an_empty_database_url(self) -> None:
        controller = SimpleNamespace(state=UiState(screen=Screen.DATABASE_URL))

        with self.assertRaisesRegex(ValueError, "Database URL cannot be empty"):
            set_database_url(controller, "  ")

    def test_existing_knowledge_bases_back_returns_to_menu(self) -> None:
        controller = SimpleNamespace(
            state=UiState(screen=Screen.EXISTING_KNOWLEDGE_BASES),
            _existing=[],
        )

        select_knowledge_base(controller, "1")

        self.assertIs(controller.state.screen, Screen.KNOWLEDGE_BASE_MENU)

    def test_database_type_back_returns_to_retrieval_type(self) -> None:
        choices = (SimpleNamespace(value="pg"), SimpleNamespace(value="back"))
        controller = SimpleNamespace(
            state=UiState(screen=Screen.DATABASE_TYPE),
            _config=SimpleNamespace(menu_items=lambda _screen: choices),
        )

        set_database_type(controller, "1")

        self.assertIs(controller.state.screen, Screen.RETRIEVAL_TYPE)
