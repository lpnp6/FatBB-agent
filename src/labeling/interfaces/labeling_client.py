"""LabelingClient — abstract interface for model backends."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..models.common import ExtractionResult


class TransientError(Exception):
    """Infrastructure error that should NOT cause permanent item rejection.

    Raised by :class:`LabelingClient` implementations when the model backend
    returns an HTTP error that is likely temporary — 404 (model loading),
    502 (bad gateway), 503 (server busy), etc.  The orchestrator catches
    this and resets the item to PENDING so it is retried on the next
    pipeline run instead of being permanently rejected.
    """

    def __init__(self, message: str, status_code: int = -1) -> None:
        super().__init__(message)
        self.status_code = status_code


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

    @abstractmethod
    async def repair(self, raw_output: str, error_message: str) -> ExtractionResult:
        """Fix a validation error in a previously-generated JSON output.

        The model receives only the broken JSON and the validator error
        message — no recipe markdown. It must return the corrected JSON.

        Args:
            raw_output: The original (invalid) JSON string.
            error_message: The validator error describing what is wrong.

        Returns:
            ExtractionResult with the corrected JSON output.
        """
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable model identifier for logging and dataset provenance.

        Examples: "gpt-4o", "local:qwen2.5-3b-fatbb-v1"
        """
        ...
