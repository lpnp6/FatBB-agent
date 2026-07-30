"""Tests for database URL handling in CLI actions."""

from types import SimpleNamespace
import unittest

from fatbb.cli.actions import set_database_url
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
