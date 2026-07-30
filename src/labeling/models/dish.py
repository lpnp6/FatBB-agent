"""Dish node and sub-structs per schema-design.md §3.1, §4, §5.3-5.4."""

from __future__ import annotations

from dataclasses import dataclass, field

from .enums import CookingMethod, Difficulty, DishType, HeatLevel, VariantType


@dataclass
class CookingStep:
    """One step in the cooking process. Order = cooking sequence."""

    order: int
    method: str  # CookingMethod slug, e.g. "stir-fry"
    method_name: str  # Human-readable, e.g. "Stir-Fry"
    ingredient_refs: list[str] = field(default_factory=list)
    note: str | None = None
    duration_min: int | None = None
    heat_level: HeatLevel | None = None


@dataclass
class CuisineRef:
    """Cuisine affiliation embedded in LLM output (not a full Cuisine node)."""

    name: str
    confidence: float  # 0.0 - 1.0
    is_primary: bool = True


@dataclass
class DishRelation:
    """A standalone Dish-to-Dish graph relationship from model output."""

    from_dish: str
    to_dish: str
    relation: str  # "variant_of" | "pairs_with"
    variant_type: VariantType | None = None
    context: str | None = None
    note: str | None = None


@dataclass
class Dish:
    """Dish node per schema-design.md §3.1."""

    # Identity
    name: str
    aliases: list[str] = field(default_factory=list)

    # Classification
    dish_type: DishType | None = None

    # Taste & Diet
    taste_profile: list[str] = field(default_factory=list)
    dietary: list[str] = field(default_factory=list)

    # Time & Difficulty
    cooking_time_min: int | None = None
    prep_time_min: int | None = None
    total_time_min: int | None = None
    difficulty: Difficulty | None = None

    # Yield & Calories
    servings: int | None = None
    calories_per_serving: int | None = None

    # Cooking Steps
    cooking_steps: list[CookingStep] = field(default_factory=list)

    # Description
    description: str | None = None
    tips: list[str] = field(default_factory=list)

    # Metadata
    source_urls: list[str] = field(default_factory=list)
    confidence: float | None = None

    # Embedded relation (resolved to an edge in the mapping layer)
    cuisine: CuisineRef | None = None
