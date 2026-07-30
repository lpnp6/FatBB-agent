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
        model: str,
        label_prompt_builder: PromptBuilder,
        repair_prompt_builder: PromptBuilder,
        base_url: str | None = None,
        max_concurrent: int = 5,
        max_tokens: int = 16384,
        empty_response_retries: int = 3,
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
        if empty_response_retries < 0:
            raise ValueError("empty_response_retries must be at least 0")

        self._label_builder = label_prompt_builder
        self._repair_builder = repair_prompt_builder
        self._model = model
        self._max_tokens = max_tokens
        self._empty_response_retries = empty_response_retries
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
        """Send one Markdown document and normalize the provider response.

        Transient empty responses from the provider are retried with exponential
        backoff before surfacing as a hard failure, so that the orchestrator
        does not waste attempt slots on API-side blips.
        """
        messages = self._label_builder.build_messages(markdown)
        async with self._semaphore:
            logger.info(
                "Dispatching labeling request model=%s input_chars=%d",
                self._model,
                len(markdown),
            )
            started_at = perf_counter()
            last_empty_error: Exception | None = None
            for attempt in range(self._empty_response_retries + 1):
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

                choices = getattr(response, "choices", None) or []
                if not choices:
                    logger.error("Labeling response had no choices model=%s", self._model)
                    raise RuntimeError("OpenAI-compatible API returned no completion choices")
                raw_output = getattr(getattr(choices[0], "message", None), "content", None)
                if isinstance(raw_output, str) and raw_output.strip():
                    break  # valid response — exit retry loop

                logger.warning(
                    "Labeling response was empty model=%s attempt=%d/%d",
                    self._model,
                    attempt + 1,
                    self._empty_response_retries + 1,
                )
                last_empty_error = RuntimeError("OpenAI-compatible API returned an empty completion")
                if attempt < self._empty_response_retries:
                    await asyncio.sleep(2 ** attempt)  # 1s, 2s, 4s, …
            else:
                logger.error("Labeling response was empty after all retries model=%s", self._model)
                raise last_empty_error  # type: ignore[arg-type]

        latency_ms = int((perf_counter() - started_at) * 1000)

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

    async def repair(self, raw_output: str, error_message: str) -> ExtractionResult:
        """Fix a validation error in a previously-generated JSON output.

        Delegates prompt assembly to the prompt builder so the repair prompt
        includes the full schema and enum reference needed for correct fixes.
        """
        messages = self._repair_builder.build_messages(raw_output, error_message)
        async with self._semaphore:
            logger.info(
                "Dispatching repair request model=%s error=%s",
                self._model,
                error_message[:120],
            )
            started_at = perf_counter()
            for attempt in range(self._empty_response_retries + 1):
                try:
                    response = await self._client.chat.completions.create(
                        model=self._model,
                        messages=messages,
                        max_tokens=self._max_tokens,
                        response_format={"type": "json_object"},
                    )
                except Exception:
                    logger.exception("Repair request failed model=%s", self._model)
                    raise

                choices = getattr(response, "choices", None) or []
                if not choices:
                    raise RuntimeError("OpenAI-compatible API returned no completion choices")
                raw = getattr(getattr(choices[0], "message", None), "content", None)
                if isinstance(raw, str) and raw.strip():
                    break
                logger.warning(
                    "Repair response was empty model=%s attempt=%d/%d",
                    self._model, attempt + 1, self._empty_response_retries + 1,
                )
                if attempt < self._empty_response_retries:
                    await asyncio.sleep(2 ** attempt)
            else:
                raise RuntimeError("OpenAI-compatible API returned an empty completion")

        latency_ms = int((perf_counter() - started_at) * 1000)
        usage = getattr(response, "usage", None)
        token_usage = TokenUsage(
            input=int(getattr(usage, "prompt_tokens", 0) or 0),
            output=int(getattr(usage, "completion_tokens", 0) or 0),
        )
        result = ExtractionResult(
            raw_output=raw,
            model=self.model_name,
            token_usage=token_usage,
            latency_ms=latency_ms,
            metadata={
                "provider_response_id": str(getattr(response, "id", "") or ""),
                "finish_reason": str(getattr(choices[0], "finish_reason", "") or ""),
                "repair": True,
            },
        )
        logger.info(
            "Repair request completed model=%s latency_ms=%d input_tokens=%d output_tokens=%d",
            result.model, result.latency_ms, result.token_usage.input, result.token_usage.output,
        )
        return result
