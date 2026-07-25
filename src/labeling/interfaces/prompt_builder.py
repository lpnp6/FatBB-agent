"""PromptBuilder — abstract interface for prompt assembly."""

from __future__ import annotations

from abc import ABC, abstractmethod


class PromptBuilder(ABC):
    """Assemble the system prompt + few-shot examples + user message.

    Abstract so the prompt strategy can evolve independently of the client:
        - Bootstrap: few-shot with hand-crafted examples, compact enum tables.
        - Production: may switch to zero-shot or different examples for the
          fine-tuned model.

    Returns messages in OpenAI chat-completions format:
        [
            {"role": "system", "content": "..."},
            {"role": "user",   "content": "..."},
            {"role": "assistant", "content": "..."},  # few-shot
            ...
            {"role": "user",   "content": "<markdown>"},
        ]
    """

    @abstractmethod
    def build_messages(self, markdown: str) -> list[dict[str, str]]:
        """Build the full message list for a single labeling call.

        Args:
            markdown: Clean, preprocessed recipe Markdown text.

        Returns:
            List of role/content dicts ready for the OpenAI chat-completions
            endpoint.
        """
        ...
