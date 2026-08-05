"""Data contracts for model evaluation.

Used by ``BaseEvaluator`` → external consumers (CLI, reporting, CI).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from labeling_sft.contracts.training import ArtifactLocation


@dataclass
class EvalReport:
    """Single-model evaluation result with domain-specific metrics as JSON."""

    model_label: str  # "Fine-tuned" | "Base" | ...
    total_examples: int
    metrics: dict[str, Any] = field(default_factory=dict)
    # Evaluator/domain-specific JSON, e.g. schema validity or task accuracy.


@dataclass
class ComparisonReport:
    """Return value of ``BaseEvaluator.compare()`` — base vs fine-tuned."""

    base_model_id: str
    adapter: ArtifactLocation
    base: EvalReport
    fine_tuned: EvalReport
    divergent_examples: list[dict[str, Any]] = field(default_factory=list)
    # Each entry: {"index": int, "base_summary": str, "ft_summary": str}
