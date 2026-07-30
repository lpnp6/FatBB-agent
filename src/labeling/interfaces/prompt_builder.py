"""PromptBuilder — abstract interface for prompt assembly."""

from __future__ import annotations

from abc import ABC, abstractmethod


class PromptBuilder(ABC):
    """Assemble messages for a model call.

    Each concrete builder specialises in one kind of prompt — labelling,
    repair, or future tasks — and exposes a single ``build_messages`` entry
    point whose signature is specific to that task.

    Returns messages in OpenAI chat-completions format:
        [
            {"role": "system", "content": "..."},
            {"role": "user",   "content": "..."},
        ]
    """

    @abstractmethod
    def build_messages(self, *args: object, **kwargs: object) -> list[dict[str, str]]:
        """Build the full message list for one model call.

        Concrete builders document their own parameter contract.
        """
        ...
