"""OpenAI-compatible asynchronous client for bootstrap labeling."""

from __future__ import annotations

import asyncio
import logging
from time import perf_counter
from typing import Any

from ..interfaces.labeling_client import LabelingClient
from ..interfaces.prompt_builder import PromptBuilder
from ..models.common import ExtractionResult, TokenUsage


logger = logging.getLogger(__name__)


class OpenAILabelingClient(LabelingClient):
    """Extract recipe data through an OpenAI-compatible Chat Completions API.

    ``base_url`` permits OpenAI-compatible providers (or a local proxy) without
    coupling the pipeline to one vendor. Responses are requested in JSON mode;
    schema validation remains the orchestrator's responsibility.

    ``client`` is an injection point for tests. In normal use it is omitted and
    an ``openai.AsyncOpenAI`` client is created lazily from ``api_key``.
    """

    def __init__(
        self,
        *,
        api_key: str,
        prompt_builder: PromptBuilder,
        model: str,
        base_url: str | None = None,
        max_concurrent: int = 5,
        max_tokens: int = 4096,
        client: Any | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("api_key must not be empty")
        if not model:
            raise ValueError("model must not be empty")
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be at least 1")
        if max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")

        self._prompt_builder = prompt_builder
        self._model = model
        self._max_tokens = max_tokens
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._client = client if client is not None else self._create_client(api_key, base_url)

    @staticmethod
    def _create_client(api_key: str, base_url: str | None) -> Any:
        try:
            from openai import AsyncOpenAI
        except ImportError as error:  # pragma: no cover - exercised in deployment setup
            raise RuntimeError(
                "The OpenAI client dependency is missing. Install the project dependencies first."
            ) from error
        return AsyncOpenAI(api_key=api_key, base_url=base_url)

    @property
    def model_name(self) -> str:
        return self._model

    async def label(self, markdown: str) -> ExtractionResult:
        """Send one Markdown document and normalize the provider response."""
        messages = self._prompt_builder.build_messages(markdown)
        async with self._semaphore:
            logger.info(
                "Dispatching labeling request model=%s input_chars=%d",
                self._model,
                len(markdown),
            )
            started_at = perf_counter()
            try:
                response = await self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    max_tokens=self._max_tokens,
                    response_format={"type": "json_object"},
                )
            except Exception:
                logger.exception("Labeling request failed model=%s", self._model)
                raise
        latency_ms = int((perf_counter() - started_at) * 1000)

        choices = getattr(response, "choices", None) or []
        if not choices:
            logger.error("Labeling response had no choices model=%s", self._model)
            raise RuntimeError("OpenAI-compatible API returned no completion choices")
        raw_output = getattr(getattr(choices[0], "message", None), "content", None)
        if not isinstance(raw_output, str) or not raw_output.strip():
            logger.error("Labeling response was empty model=%s", self._model)
            raise RuntimeError("OpenAI-compatible API returned an empty completion")

        usage = getattr(response, "usage", None)
        token_usage = TokenUsage(
            input=int(getattr(usage, "prompt_tokens", 0) or 0),
            output=int(getattr(usage, "completion_tokens", 0) or 0),
        )
        result = ExtractionResult(
            raw_output=raw_output,
            model=self.model_name,
            token_usage=token_usage,
            latency_ms=latency_ms,
            metadata={
                "provider_response_id": str(getattr(response, "id", "") or ""),
                "finish_reason": str(getattr(choices[0], "finish_reason", "") or ""),
            },
        )
        logger.info(
            "Labeling request completed model=%s latency_ms=%d input_tokens=%d "
            "output_tokens=%d response_id=%s finish_reason=%s",
            result.model,
            result.latency_ms,
            result.token_usage.input,
            result.token_usage.output,
            result.metadata["provider_response_id"],
            result.metadata["finish_reason"],
        )
        return result
