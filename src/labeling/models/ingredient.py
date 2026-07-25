"""Ingredient node and relation schemas per schema-design.md §3.2, §5.1, §5.5."""

from __future__ import annotations

from dataclasses import dataclass, field

from .enums import (
    ComplementStrength,
    IngredientCategory,
    StorageType,
    SubstituteDirection,
    SubstituteImpact,
    UmamiLevel,
)


@dataclass
class AmountNormalized:
    """Normalized quantity for a CONTAINS relationship."""

    value: float
    unit: str
    range_low: float | None = None
    range_high: float | None = None


@dataclass
class Ingredient:
    """Ingredient node per schema-design.md §3.2."""

    # Identity
    name: str
    aliases: list[str] = field(default_factory=list)

    # Classification
    category: IngredientCategory | None = None
    sub_category: str | None = None

    # Inline CONTAINS relationship fields (embedded in LLM output)
    amount: str = ""  # Original text: "200g", "to taste"
    amount_normalized: AmountNormalized | None = None
    is_essential: bool = True
    preparation: str | None = None  # "diced", "minced", "pre-soaked"
    notes: str | None = None  # "can substitute with XX", "optional"

    # Nutrition (per 100g)
    calories_per_100g: float | None = None
    protein_g: float | None = None
    fat_g: float | None = None
    carbs_g: float | None = None
    fiber_g: float | None = None
    sodium_mg: float | None = None

    # Characteristics
    storage: StorageType | None = None
    umami_level: UmamiLevel | None = None
    season: list[str] = field(default_factory=list)

    # Description
    description: str | None = None

    # Metadata
    source_urls: list[str] = field(default_factory=list)


@dataclass
class IngredientRelation:
    """An ingredient-to-ingredient relationship (COMPLEMENTS, SUBSTITUTES, MAKES).

    Per schema-design.md §5.5 and §8 ingredient_relations format.
    """

    from_ingredient: str  # Ingredient name or slug
    to_ingredient: str  # Ingredient name or slug
    relation: str  # "complements" | "substitutes" | "makes"

    # COMPLEMENTS fields
    strength: ComplementStrength | None = None
    context: str | None = None

    # SUBSTITUTES fields
    direction: SubstituteDirection | None = None
    impact: SubstituteImpact | None = None
    condition: str | None = None

    # MAKES fields
    process: str | None = None  # e.g. "soaking→grinding→boiling→coagulation→molding"
    is_reversible: bool = False

    # Shared
    note: str | None = None
