"""Abstract base class for model evaluators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from labeling_sft.contracts import (
    ArtifactLocation,
    ComparisonReport,
    DatasetSplit,
    EvalReport,
    TrainingResult,
)


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
        training: TrainingResult,
        include_adapter: bool = True,
        **kwargs,
    ) -> tuple[Any, Any]:
        """Load model and tokenizer.

        Args:
            training: Model artifacts and base model identity from training.
            include_adapter: Whether to apply the fine-tuned adapter.

        Returns:
            ``(model, tokenizer)``
        """
        ...

    @abstractmethod
    def evaluate(
        self,
        training: TrainingResult,
        dataset: DatasetSplit,
        report_target: ArtifactLocation | None = None,
        max_samples: int | None = None,
        **kwargs,
    ) -> EvalReport:
        """Evaluate a single model.

        Args:
            training:      Model artifacts produced by a trainer.
            dataset:       Validation split produced by a dataset builder.
            report_target: Where to write the report JSON (``None`` = skip).
            max_samples:   Limit to first N samples (``None`` = all).

        Returns:
            :class:`~labeling_sft.interfaces.contracts.EvalReport`.
        """
        ...

    @abstractmethod
    def compare(
        self,
        training: TrainingResult,
        dataset: DatasetSplit,
        report_target: ArtifactLocation | None = None,
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
