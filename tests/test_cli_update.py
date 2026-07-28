import unittest

from fatbb.cli.events import InputChanged, KeyPressed
from fatbb.cli.state import Screen, UiState
from fatbb.cli.update import update


class CliUpdateTests(unittest.TestCase):
    def test_slash_opens_command_palette(self) -> None:
        transition = update(UiState(), InputChanged("/"))

        self.assertIs(transition.state.screen, Screen.PALETTE)
        self.assertEqual(transition.state.input_text, "/")

    def test_deleting_slash_returns_to_chat(self) -> None:
        state = UiState(screen=Screen.PALETTE, input_text="/", active_knowledge_base_id="kb-1")
        transition = update(state, InputChanged(""))

        self.assertIs(transition.state.screen, Screen.CHAT)
        self.assertEqual(transition.state.active_knowledge_base_id, "kb-1")

    def test_palette_enter_opens_knowledge_base_menu(self) -> None:
        transition = update(UiState(screen=Screen.PALETTE, input_text="/"), KeyPressed("enter"))

        self.assertIs(transition.state.screen, Screen.KNOWLEDGE_BASE_MENU)
        self.assertEqual(transition.state.input_text, "")

    def test_arrow_navigation_wraps(self) -> None:
        transition = update(UiState(screen=Screen.KNOWLEDGE_BASE_MENU), KeyPressed("up"), item_count=3)

        self.assertEqual(transition.state.selected_index, 2)
