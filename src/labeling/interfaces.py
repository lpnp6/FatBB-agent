"""Abstract interfaces for the labeling pipeline.

Every interface here follows dependency inversion: the orchestrator imports
ONLY these ABCs. Concrete implementations are injected at the composition root
(run.py), enabling backend swaps with zero pipeline code changes.

Interface summary:
    LabelingClient  — call a model to extract structured food KG data
    DedupStore      — persist recipe-card hashes; block duplicates before API calls
    PromptBuilder   — assemble system + few-shot + user messages for the model
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from .models.common import ExtractionResult


# =============================================================================
# LabelingClient
# =============================================================================


class LabelingClient(ABC):
    """Call a model to extract structured food KG data from recipe markdown.

    Abstract over the model backend. Two planned implementations:
        - OpenAILabelingClient  (bootstrap, OpenAI-compatible cloud API)
        - LocalModelLabelingClient (production, fine-tuned Qwen on local GPU)

    The orchestrator calls label() once per file and gets back a raw
    ExtractionResult. Validation, scoring, and persistence happen downstream.
    """

    @abstractmethod
    async def label(self, markdown: str) -> ExtractionResult:
        """Extract structured food KG data from a single recipe's markdown.

        Args:
            markdown: Clean, preprocessed recipe Markdown text.

        Returns:
            ExtractionResult with the model's raw JSON output, token usage,
            and timing metadata.
        """
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable model identifier for logging and dataset provenance.

        Examples: "gpt-4o", "local:qwen2.5-3b-fatbb-v1"
        """
        ...


# =============================================================================
# DedupStore
# =============================================================================


class DedupStore(ABC):
    """Persistent store of recipe-card content hashes.

    Two responsibilities:
        1. During initial sampling: group duplicate recipe files (wprm_print vs
           full_page variants of the same recipe) so only one is selected.
        2. During labeling: check each file's hash BEFORE calling the model.
           If the hash is already registered, skip without incurring API cost.

    Two planned implementations:
        - SQLiteDedupStore  (persistent, survives process restarts)
        - MemoryDedupStore  (in-memory set, for unit tests, lost on restart)
    """

    @abstractmethod
    def is_duplicate(self, recipe_card_hash: str) -> bool:
        """Return True if this hash has already been registered.

        O(1) lookup. Called before every client.label() call.
        """
        ...

    @abstractmethod
    def register(
        self, recipe_card_hash: str, source_file: str, variant: str
    ) -> None:
        """Persist a hash after a file is accepted for labeling.

        Called after client.label() succeeds, so the recipe is blocked from
        being re-labeled in future runs.

        Args:
            recipe_card_hash: SHA-256 fingerprint of the recipe card portion.
            source_file: Original file path or slug for provenance.
            variant: "wprm_print" | "full_page" — which variant was labeled.
        """
        ...

    @abstractmethod
    def recipe_card_hash(self, markdown: str) -> str:
        """Compute a stable SHA-256 fingerprint of the recipe card.

        Extracts the ### Ingredients + ### Instructions blocks from the
        markdown, normalizes whitespace, then SHA-256 hashes the result.
        This is deliberately NOT a semantic hash — it only matches
        byte-identical recipe cards (same recipe, same site, different URL
        variant).

        Args:
            markdown: Full recipe Markdown text (may include intro, comments).

        Returns:
            64-character hex SHA-256 digest.
        """
        ...


# =============================================================================
# PromptBuilder
# =============================================================================


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
