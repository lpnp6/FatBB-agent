"""Tests for database URL handling in CLI actions."""

from types import SimpleNamespace
import unittest

from fatbb.cli.actions import (
    select_knowledge_base,
    set_database_type,
    set_database_url,
    set_embedding_model,
    set_embedding_provider,
    set_embedding_url,
)
from fatbb.cli.state import UiState


class SetDatabaseUrlTests(unittest.TestCase):
    @staticmethod
    def _config() -> SimpleNamespace:
        routes = {
            "database_url_next": "source_type",
            "existing_knowledge_bases_back": "knowledge_base_menu",
            "database_type": "database_type",
            "database_type_back": "retrieval_type",
        }
        return SimpleNamespace(
            route=lambda name: routes[name],
            menu_items=lambda _screen: (SimpleNamespace(value="pg"), SimpleNamespace(value="back")),
        )

    @staticmethod
    def _vector_config() -> SimpleNamespace:
        routes = {
            "database_url_vector_next": "embedding_provider",
            "embedding_provider": "embedding_provider",
            "embedding_provider_next": "embedding_model",
            "embedding_provider_back": "database_url",
            "embedding_model": "embedding_model",
            "embedding_model_back": "embedding_provider",
            "embedding_model_next": "embedding_url",
            "embedding_url_next": "source_type",
        }
        return SimpleNamespace(
            route=lambda name: routes[name],
            menu_items=lambda screen: (
                (SimpleNamespace(value="nomic-embed-text"), SimpleNamespace(value="back"))
                if screen == "embedding_model"
                else (SimpleNamespace(value="ollama"), SimpleNamespace(value="back"))
            ),
        )

    def test_extracts_a_non_postgresql_database_url_from_a_multiline_paste(self) -> None:
        controller = SimpleNamespace(state=UiState(screen="database_url"), _config=self._config())

        set_database_url(controller, "database name\nmongodb://localhost:27017/fatbb\n")

        self.assertEqual(
            controller.state.pending_database_url,
            "mongodb://localhost:27017/fatbb",
        )
        self.assertEqual(controller.state.status, "URL extracted from multi-line paste.")

    def test_rejects_an_empty_database_url(self) -> None:
        controller = SimpleNamespace(state=UiState(screen="database_url"), _config=self._config())

        with self.assertRaisesRegex(ValueError, "Database URL cannot be empty"):
            set_database_url(controller, "  ")

    def test_vector_database_url_advances_to_embedding_provider(self) -> None:
        controller = SimpleNamespace(
            state=UiState(screen="database_url", pending_retrieval_type="vector"),
            _config=self._vector_config(),
        )

        set_database_url(controller, "postgresql://localhost/fatbb")

        self.assertEqual(controller.state.screen, "embedding_provider")

    def test_stores_selected_ollama_provider_model_and_url(self) -> None:
        controller = SimpleNamespace(
            state=UiState(screen="embedding_provider"), _config=self._vector_config()
        )

        set_embedding_provider(controller, "0")
        self.assertEqual(controller.state.pending_embedding_provider, "ollama")
        self.assertEqual(controller.state.screen, "embedding_model")

        set_embedding_model(controller, "0")
        self.assertEqual(controller.state.pending_embedding_model, "nomic-embed-text")
        self.assertEqual(controller.state.screen, "embedding_url")

        set_embedding_url(controller, "http://localhost:11434")
        self.assertEqual(controller.state.pending_embedding_url, "http://localhost:11434")
        self.assertEqual(controller.state.screen, "source_type")

    def test_existing_knowledge_bases_back_returns_to_menu(self) -> None:
        controller = SimpleNamespace(
            state=UiState(screen="existing_knowledge_bases"),
            _existing=[],
            _config=self._config(),
        )

        select_knowledge_base(controller, "1")

        self.assertEqual(controller.state.screen, "knowledge_base_menu")

    def test_database_type_back_returns_to_retrieval_type(self) -> None:
        controller = SimpleNamespace(
            state=UiState(screen="database_type"),
            _config=self._config(),
        )

        set_database_type(controller, "1")

        self.assertEqual(controller.state.screen, "retrieval_type")
