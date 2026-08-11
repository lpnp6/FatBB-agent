"""Ollama asynchronous client for local inference with fine-tuned models."""

from __future__ import annotations

import asyncio
import logging
from time import perf_counter
from typing import Any

from ..interfaces.labeling_client import LabelingClient, TransientError
from ..interfaces.prompt_builder import PromptBuilder
from ..models.common import ExtractionResult, TokenUsage

# HTTP status codes that indicate a transient infrastructure issue rather
# than a permanent data problem.  Items that fail with these codes should
# be reset to PENDING and retried on the next pipeline run.
_TRANSIENT_HTTP_CODES: frozenset[int] = frozenset({404, 502, 503, 504})

logger = logging.getLogger(__name__)


def _wrap_transient(exc: Exception) -> None:
    """If *exc* is an HTTP error with a transient status code, raise
    :class:`TransientError` so the orchestrator resets the item to PENDING
    instead of permanently rejecting it.

    Uses duck-typing on ``status_code`` so the module does not need a
    hard import of ``ollama`` at the top level.
    """
    status_code: int = getattr(exc, "status_code", -1)
    if status_code in _TRANSIENT_HTTP_CODES:
        raise TransientError(str(exc), status_code=status_code) from exc


def _val(obj: Any, field: str, default: Any = None) -> Any:
    """Read *field* from *obj*, supporting both attribute and dict access.

    The Ollama Python library returns Pydantic ``ChatResponse`` models in
    normal use, but tests may inject plain dicts — this helper handles both.
    """
    if obj is None:
        return default
    if hasattr(obj, field):
        return getattr(obj, field, default)
    if isinstance(obj, dict):
        return obj.get(field, default)
    return default


class OllamaLabelingClient(LabelingClient):
    """Extract recipe data through a local Ollama server.

    Talks to Ollama's chat API (``/api/chat``).  ``host`` defaults to the
    standard local address; set ``OLLAMA_HOST`` or pass *host* explicitly to
    point at a remote instance.

    Injection points (``client``) allow tests to swap in a fake without
    touching the real server.
    """

    def __init__(
        self,
        *,
        model: str,
        label_prompt_builder: PromptBuilder,
        repair_prompt_builder: PromptBuilder,
        host: str = "http://127.0.0.1:11434",
        max_concurrent: int = 1,
        num_ctx: int = 32768,
        num_predict: int = 16384,
        temperature: float = 0.1,
        empty_response_retries: int = 3,
        client: Any | None = None,
    ) -> None:
        if not model:
            raise ValueError("model must not be empty")
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be at least 1")
        if num_ctx < 1:
            raise ValueError("num_ctx must be at least 1")
        if num_predict < 1:
            raise ValueError("num_predict must be at least 1")
        if empty_response_retries < 0:
            raise ValueError("empty_response_retries must be at least 0")
        if not (0.0 <= temperature <= 2.0):
            raise ValueError("temperature must be in [0.0, 2.0]")

        self._model = model
        self._label_builder = label_prompt_builder
        self._repair_builder = repair_prompt_builder
        self._num_ctx = num_ctx
        self._num_predict = num_predict
        self._temperature = temperature
        self._empty_response_retries = empty_response_retries
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._client = client if client is not None else self._create_client(host)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @staticmethod
    def _create_client(host: str) -> Any:
        try:
            import ollama  # noqa: F401 — validate package presence
        except ImportError as error:
            raise RuntimeError(
                "The ollama Python package is required for OllamaLabelingClient. "
                "Install it with: pip install ollama"
            ) from error
        return ollama.AsyncClient(host=host)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        return self._model

    # ------------------------------------------------------------------
    # label
    # ------------------------------------------------------------------

    async def label(self, markdown: str) -> ExtractionResult:
        """Send one Markdown document to Ollama and normalise the response."""
        messages = self._label_builder.build_messages(markdown)

        async with self._semaphore:
            logger.info(
                "Dispatching labeling request model=%s input_chars=%d",
                self._model, len(markdown),
            )
            started_at = perf_counter()
            last_empty_error: Exception | None = None

            for attempt in range(self._empty_response_retries + 1):
                try:
                    response = await self._client.chat(
                        model=self._model,
                        messages=messages,
                        stream=False,
                        options={
                            "num_ctx": self._num_ctx,
                            "num_predict": self._num_predict,
                            "temperature": self._temperature,
                        },
                    )
                except Exception as exc:
                    _wrap_transient(exc)
                    logger.exception("Labeling request failed model=%s", self._model)
                    raise

                raw_output = _val(_val(response, "message"), "content")
                if isinstance(raw_output, str) and raw_output.strip():
                    break

                logger.warning(
                    "Labeling response was empty model=%s attempt=%d/%d",
                    self._model, attempt + 1, self._empty_response_retries + 1,
                )
                last_empty_error = RuntimeError("Ollama returned an empty completion")
                if attempt < self._empty_response_retries:
                    await asyncio.sleep(2 ** attempt)
            else:
                logger.error(
                    "Labeling response was empty after all retries model=%s",
                    self._model,
                )
                raise last_empty_error  # type: ignore[arg-type]

        latency_ms = int((perf_counter() - started_at) * 1000)

        token_usage = TokenUsage(
            input=int(_val(response, "prompt_eval_count", 0) or 0),
            output=int(_val(response, "eval_count", 0) or 0),
        )
        result = ExtractionResult(
            raw_output=raw_output,  # type: ignore[arg-type]
            model=self.model_name,
            token_usage=token_usage,
            latency_ms=latency_ms,
            metadata={
                "done_reason": str(_val(response, "done_reason", "") or ""),
                "total_duration_ns": int(_val(response, "total_duration", 0) or 0),
                "load_duration_ns": int(_val(response, "load_duration", 0) or 0),
            },
        )
        logger.info(
            "Labeling request completed model=%s latency_ms=%d input_tokens=%d "
            "output_tokens=%d done_reason=%s",
            result.model, result.latency_ms,
            result.token_usage.input, result.token_usage.output,
            result.metadata["done_reason"],
        )
        return result

    # ------------------------------------------------------------------
    # repair
    # ------------------------------------------------------------------

    async def repair(self, raw_output: str, error_message: str) -> ExtractionResult:
        """Fix a validation error in a previously-generated JSON output."""
        messages = self._repair_builder.build_messages(raw_output, error_message)

        async with self._semaphore:
            logger.info(
                "Dispatching repair request model=%s error=%s",
                self._model, error_message[:120],
            )
            started_at = perf_counter()
            last_empty_error: Exception | None = None

            for attempt in range(self._empty_response_retries + 1):
                try:
                    response = await self._client.chat(
                        model=self._model,
                        messages=messages,
                        stream=False,
                        options={
                            "num_ctx": self._num_ctx,
                            "num_predict": self._num_predict,
                            "temperature": self._temperature,
                        },
                    )
                except Exception as exc:
                    _wrap_transient(exc)
                    logger.exception("Repair request failed model=%s", self._model)
                    raise

                repaired = _val(_val(response, "message"), "content")
                if isinstance(repaired, str) and repaired.strip():
                    break

                logger.warning(
                    "Repair response was empty model=%s attempt=%d/%d",
                    self._model, attempt + 1, self._empty_response_retries + 1,
                )
                last_empty_error = RuntimeError("Ollama returned an empty completion")
                if attempt < self._empty_response_retries:
                    await asyncio.sleep(2 ** attempt)
            else:
                raise last_empty_error  # type: ignore[arg-type]

        latency_ms = int((perf_counter() - started_at) * 1000)

        token_usage = TokenUsage(
            input=int(_val(response, "prompt_eval_count", 0) or 0),
            output=int(_val(response, "eval_count", 0) or 0),
        )
        result = ExtractionResult(
            raw_output=repaired,  # type: ignore[arg-type]
            model=self.model_name,
            token_usage=token_usage,
            latency_ms=latency_ms,
            metadata={
                "done_reason": str(_val(response, "done_reason", "") or ""),
                "repair": True,
            },
        )
        logger.info(
            "Repair request completed model=%s latency_ms=%d input_tokens=%d output_tokens=%d",
            result.model, result.latency_ms,
            result.token_usage.input, result.token_usage.output,
        )
        return result
