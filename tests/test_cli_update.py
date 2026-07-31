import unittest

from fatbb.cli.events import InputChanged, KeyPressed
from fatbb.cli.config import Page
from fatbb.cli.state import UiState
from fatbb.cli.update import update


class CliUpdateTests(unittest.TestCase):
    def _update(self, state: UiState, event: InputChanged | KeyPressed, *, page: Page, item_count: int = 1):
        return update(state, event, page=page, home_page="chat", palette_page="palette", item_count=item_count)

    def test_slash_opens_command_palette(self) -> None:
        transition = self._update(UiState(screen="chat"), InputChanged("/"), page=Page("chat", "retrieve"))

        self.assertEqual(transition.state.screen, "palette")
        self.assertEqual(transition.state.input_text, "/")

    def test_deleting_slash_returns_to_chat(self) -> None:
        state = UiState(screen="palette", input_text="/", active_knowledge_base_id="kb-1")
        transition = self._update(state, InputChanged(""), page=Page("menu", "palette_selection"))

        self.assertEqual(transition.state.screen, "chat")
        self.assertEqual(transition.state.active_knowledge_base_id, "kb-1")

    def test_palette_enter_opens_knowledge_base_menu(self) -> None:
        transition = self._update(
            UiState(screen="palette", input_text="/"), KeyPressed("enter"),
            page=Page("menu", "palette_selection"),
        )

        self.assertEqual(transition.state.screen, "palette")
        self.assertEqual(transition.action.kind, "palette_selection")

    def test_palette_back_returns_to_chat(self) -> None:
        transition = self._update(
            UiState(screen="palette", input_text="/", selected_index=1), KeyPressed("enter"),
            page=Page("menu", "palette_selection"),
            item_count=2,
        )

        self.assertEqual(transition.action.kind, "palette_selection")
        self.assertEqual(transition.action.value, "1")

    def test_arrow_navigation_wraps(self) -> None:
        transition = self._update(UiState(screen="knowledge_base_menu"), KeyPressed("up"), page=Page("menu"), item_count=3)

        self.assertEqual(transition.state.selected_index, 2)

    def test_database_type_emits_configured_selection_action(self) -> None:
        transition = self._update(UiState(screen="database_type"), KeyPressed("enter"), page=Page("menu", "set_database_type"))

        self.assertEqual(transition.state.screen, "database_type")
        self.assertIsNotNone(transition.action)
        self.assertEqual(transition.action.kind, "set_database_type")

    def test_progress_page_ignores_input_and_navigation(self) -> None:
        state = UiState(screen="indexing")
        page = Page("progress")

        for event in (InputChanged("cancel"), KeyPressed("escape"), KeyPressed("enter")):
            transition = self._update(state, event, page=page)
            self.assertEqual(transition.state, state)
            self.assertIsNone(transition.action)
