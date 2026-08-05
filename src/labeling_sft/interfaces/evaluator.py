"""Abstract base class for model evaluators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from labeling_sft.contracts import ComparisonReport, EvalReport


class BaseEvaluator(ABC):
    """Evaluate a fine-tuned model against a validation set.

    Supports two evaluation modes:

    * :meth:`evaluate`  — single-model evaluation
    * :meth:`compare`   — base model vs fine-tuned model side-by-side

    Concrete implementations:
        - :class:`~labeling_sft.evaluators.qwen.QwenEvaluator`
        - Future: ``GenericEvaluator``, ``BatchEvaluator``, ...
    """

    @abstractmethod
    def load_model(
        self,
        adapter_dir: str | None,
        base_model_id: str,
        **kwargs,
    ) -> tuple[Any, Any]:
        """Load model and tokenizer.

        Args:
            adapter_dir:  LoRA adapter path. ``None`` = load base model only
                          (used for comparison mode).
            base_model_id: HuggingFace model ID.

        Returns:
            ``(model, tokenizer)``
        """
        ...

    @abstractmethod
    def evaluate(
        self,
        adapter_dir: str | None,
        val_path: str,
        base_model_id: str,
        output_report: str | None = None,
        max_samples: int | None = None,
        **kwargs,
    ) -> EvalReport:
        """Evaluate a single model.

        Args:
            adapter_dir:   Adapter path (``None`` = base model only).
            val_path:      Path to ``val.jsonl``.
            base_model_id: Base model HuggingFace ID.
            output_report: Where to write the report JSON (``None`` = skip).
            max_samples:   Limit to first N samples (``None`` = all).

        Returns:
            :class:`~labeling_sft.interfaces.contracts.EvalReport`.
        """
        ...

    @abstractmethod
    def compare(
        self,
        adapter_dir: str,
        val_path: str,
        base_model_id: str,
        output_report: str | None = None,
        diff_examples: int = 5,
        max_samples: int | None = None,
        **kwargs,
    ) -> ComparisonReport:
        """Compare base model vs fine-tuned model.

        Runs the base model first, frees GPU memory, then runs the fine-tuned
        model, and produces a side-by-side comparison report.

        Returns:
            :class:`~labeling_sft.interfaces.contracts.ComparisonReport`.
        """
        ...
