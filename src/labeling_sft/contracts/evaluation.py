"""Data contracts for model evaluation.

Used by ``BaseEvaluator`` → external consumers (CLI, reporting, CI).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvalReport:
    """Single-model evaluation result."""

    model_label: str  # "Fine-tuned" | "Base" | ...
    total_examples: int
    json_valid: int
    json_validity_pct: float
    validator_pass: int
    validator_pass_pct: float
    enum_valid_fields: int
    enum_total_fields: int
    enum_accuracy_pct: float
    not_a_recipe_correct: int
    not_a_recipe_total: int
    not_a_recipe_accuracy_pct: float | None
    field_coverage: dict[str, dict[str, int]] = field(default_factory=dict)
    # {field: {present, total, pct}}
    validator_errors: dict[str, int] = field(default_factory=dict)
    # {error_type: count}
    raw_metrics: dict[str, Any] = field(default_factory=dict)
    # Complete raw metrics (for debugging / serialization)


@dataclass
class ComparisonReport:
    """Return value of ``BaseEvaluator.compare()`` — base vs fine-tuned."""

    base_model_id: str
    adapter_dir: str
    base: EvalReport
    fine_tuned: EvalReport
    divergent_examples: list[dict[str, Any]] = field(default_factory=list)
    # Each entry: {"index": int, "base_summary": str, "ft_summary": str}
