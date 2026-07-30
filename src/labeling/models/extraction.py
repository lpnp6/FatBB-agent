"""Pipeline output schemas: what the labeling pipeline produces and stores."""

from __future__ import annotations

from dataclasses import dataclass, field

from .common import ExtractionResult, TokenUsage
from .dish import Dish, DishRelation
from .ingredient import Ingredient, IngredientRelation


@dataclass
class ExtractionOutput:
    """Full structured output from one recipe labeling call.

    Matches schema-design.md §8 LLM Extraction Output Format.
    """

    dish: Dish | None = None
    """Extracted dish node. Null if the file is not a recipe."""

    ingredients: list[Ingredient] = field(default_factory=list)
    """Ingredients with inline CONTAINS relationship fields."""

    ingredient_relations: list[IngredientRelation] = field(default_factory=list)
    """Inter-ingredient relationships (complements, substitutes, makes)."""

    dish_relations: list[DishRelation] = field(default_factory=list)
    """Dish relationships (variant_of, pairs_with) outside the Dish node."""


@dataclass
class ValidationError:
    """One validation failure for a labeled output."""

    field: str  # JSON path, e.g. "dish.name" or "ingredients[2].category"
    code: str  # "missing_required" | "invalid_enum" | "cross_ref_mismatch" | ...
    message: str


@dataclass
class ValidationResult:
    """Result of validating one ExtractionOutput."""

    is_valid: bool
    errors: list[ValidationError] = field(default_factory=list)
    warnings: list[ValidationError] = field(default_factory=list)


@dataclass
class LabelResult:
    """Complete result of labeling one file — raw + parsed + validation + scoring."""

    source_slug: str
    """Identifier derived from the source file name."""

    source_path: str
    """Original markdown file path."""

    extraction: ExtractionResult
    """Raw model output + token usage."""

    parsed: ExtractionOutput | None = None
    """Parsed and validated structured output. Null if JSON parse failed."""

    validation: ValidationResult = field(default_factory=ValidationResult)

    confidence: float = 0.0
    """0.0 - 1.0 confidence score from scorer.py."""

    needs_review: bool = False
    """True if confidence below threshold — human should inspect."""

    batch: str = ""
    """Labeling batch identifier (e.g. "batch_001")."""

    retries: int = 0
    """Number of retries attempted for this file."""

    error: str | None = None
    """Fatal error message if labeling failed completely."""
