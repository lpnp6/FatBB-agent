"""Coordinates pure UI transitions with application use cases."""

from __future__ import annotations

from dataclasses import replace

from fatbb.application.knowledge_base_service import KnowledgeBaseService
from fatbb.domain.knowledge_base import KnowledgeBase

from .events import InputChanged, KeyPressed
from .state import Screen, UiState
from .update import Transition, update


class CliController:
    def __init__(self, service: KnowledgeBaseService):
        self._service = service
        self.state = UiState()
        self._existing: list[KnowledgeBase] = []

    def items(self) -> tuple[str, ...]:
        match self.state.screen:
            case Screen.PALETTE:
                return ("Knowledge Base",)
            case Screen.KNOWLEDGE_BASE_MENU:
                return ("Select existing", "Create new", "Back")
            case Screen.EXISTING_KNOWLEDGE_BASES:
                return tuple(kb.name for kb in self._existing) or ("No knowledge bases found",)
            case Screen.RETRIEVAL_TYPE:
                return ("BM25",)
            case Screen.DATABASE_TYPE:
                return ("PostgreSQL",)
            case Screen.SOURCE_TYPE:
                return ("File path",)
            case _:
                return ()

    def on_input_changed(self, text: str) -> None:
        self._apply(update(self.state, InputChanged(text), item_count=len(self.items())))

    def on_key_pressed(self, key: str) -> None:
        if key == "ctrl_d":
            raise EOFError
        self._apply(update(self.state, KeyPressed(key), item_count=len(self.items())))

    def _apply(self, transition: Transition) -> None:
        self.state = transition.state
        if transition.action is not None:
            self._run(transition.action.kind, transition.action.value)

    def _run(self, kind: str, value: str | None) -> None:
        try:
            if kind == "knowledge_base_menu_selection":
                self._knowledge_base_menu(int(value or "0"))
            elif kind == "select_knowledge_base":
                self._select_existing(int(value or "0"))
            elif kind == "set_knowledge_base_name":
                self._set_name(value or "")
            elif kind == "create_knowledge_base":
                self._create(value or "")
            elif kind == "retrieve":
                self._retrieve(value or "")
        except (RuntimeError, ValueError) as error:
            self.state = replace(self.state, status=f"Error: {error}")

    def _knowledge_base_menu(self, index: int) -> None:
        if index == 0:
            self._existing = self._service.list()
            self.state = replace(self.state, screen=Screen.EXISTING_KNOWLEDGE_BASES, selected_index=0)
        elif index == 1:
            self.state = replace(self.state, screen=Screen.RETRIEVAL_TYPE, selected_index=0)
        else:
            self.state = replace(self.state, screen=Screen.CHAT, selected_index=0)

    def _select_existing(self, index: int) -> None:
        if not self._existing:
            self.state = replace(self.state, status="No knowledge bases are available.")
            return
        knowledge_base = self._existing[index]
        self._activate(knowledge_base, "Knowledge base selected.")

    def _set_name(self, name: str) -> None:
        if not name.strip():
            raise ValueError("Knowledge base name cannot be empty.")
        self.state = replace(
            self.state, screen=Screen.SOURCE_PATH, input_text="", pending_name=name.strip(),
            status="Enter a local file or directory path.",
        )

    def _create(self, source_path: str) -> None:
        knowledge_base = self._service.create(self.state.pending_name, source_path.strip())
        self._activate(knowledge_base, f'Indexed and selected "{knowledge_base.name}".')

    def _retrieve(self, question: str) -> None:
        if not self.state.active_knowledge_base_id:
            self.state = replace(self.state, input_text="", status="Select a knowledge base with / first.")
            return
        knowledge_base = next(
            kb for kb in self._service.list() if kb.id == self.state.active_knowledge_base_id
        )
        evidence = self._service.retrieve(knowledge_base, question)
        lines = tuple(_format_evidence(item, index + 1) for index, item in enumerate(evidence))
        self.state = replace(
            self.state, input_text="", lines=lines,
            status=(f"Retrieved {len(evidence)} relevant sources." if evidence else "No relevant sources found."),
        )

    def _activate(self, knowledge_base: KnowledgeBase, status: str) -> None:
        self.state = replace(
            self.state, screen=Screen.CHAT, input_text="", selected_index=0,
            active_knowledge_base_id=knowledge_base.id,
            active_knowledge_base_name=knowledge_base.name, status=status, lines=(),
        )


def _format_evidence(item: object, index: int) -> str:
    from rag.models.evidence import Evidence

    evidence = item if isinstance(item, Evidence) else None
    if evidence is None:
        return str(item)
    source = evidence.source.title if evidence.source and evidence.source.title else "Unknown source"
    content = " ".join(evidence.content.split())
    preview = content[:300] + ("…" if len(content) > 300 else "")
    return f"{index}. {source} · score {evidence.score:.2f}\n   {preview}"
