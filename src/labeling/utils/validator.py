"""Parsing and validation of the labeling prompt's JSON contract."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from ..models import (
    AmountNormalized, ComplementStrength, CookingMethod, CookingStep, CuisineRef,
    DietaryTag, Difficulty, Dish, DishRelation, DishType, ExtractionOutput, HeatLevel,
    Ingredient, IngredientCategory, IngredientRelation, SubstituteDirection,
    SubstituteImpact, TasteProfile, VariantType,
)


@dataclass(frozen=True)
class ParsedLabel:
    output: ExtractionOutput | None
    is_not_a_recipe: bool
    normalized_json: dict[str, Any]


class OutputValidationError(ValueError):
    """The model returned JSON that does not match the labeling contract."""


def _enum(enum_type: type, value: Any, field: str) -> Any:
    if value is None:
        return None
    try:
        return enum_type(value)
    except ValueError as error:
        raise OutputValidationError(f"{field}: invalid value {value!r}") from error


def _mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise OutputValidationError(f"{field}: expected object")
    return value


def _list(value: Any, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise OutputValidationError(f"{field}: expected list")
    return value


def _string(value: Any, field: str, *, required: bool = False) -> str | None:
    if value is None and not required:
        return None
    if not isinstance(value, str) or (required and not value.strip()):
        raise OutputValidationError(f"{field}: expected non-empty string")
    return value


class OutputValidator:
    """Convert one JSON response into the dataclass contract used by the pipeline."""

    def parse(self, raw_output: str) -> ParsedLabel:
        try:
            payload = json.loads(raw_output)
        except json.JSONDecodeError as error:
            raise OutputValidationError(f"invalid JSON: {error.msg}") from error
        payload = _mapping(payload, "root")

        if payload.get("dish") is None:
            if payload.get("reason") != "not_a_recipe":
                raise OutputValidationError("non-recipe output must use reason=not_a_recipe")
            if payload.get("ingredients") != [] or payload.get("ingredient_relations") != []:
                raise OutputValidationError("non-recipe output must have empty ingredient lists")
            return ParsedLabel(output=None, is_not_a_recipe=True, normalized_json=payload)

        dish = self._dish(_mapping(payload["dish"], "dish"))
        ingredients = [self._ingredient(_mapping(value, "ingredients[]")) for value in _list(payload.get("ingredients"), "ingredients")]
        ingredient_relations = [
            self._ingredient_relation(_mapping(value, "ingredient_relations[]"))
            for value in _list(payload.get("ingredient_relations"), "ingredient_relations")
        ]
        dish_relations = [
            self._dish_relation(_mapping(value, "dish_relations[]"))
            for value in _list(payload.get("dish_relations"), "dish_relations")
        ]
        ingredient_slugs = {ingredient.name.lower().replace(" ", "-") for ingredient in ingredients}
        for step in dish.cooking_steps:
            unknown = set(step.ingredient_refs) - ingredient_slugs
            if unknown:
                raise OutputValidationError(f"cooking_steps ingredient_refs not found: {sorted(unknown)!r}")
        return ParsedLabel(
            output=ExtractionOutput(dish=dish, ingredients=ingredients, ingredient_relations=ingredient_relations, dish_relations=dish_relations),
            is_not_a_recipe=False,
            normalized_json=payload,
        )

    def _dish(self, value: dict[str, Any]) -> Dish:
        steps = []
        for step in _list(value.get("cooking_steps"), "dish.cooking_steps"):
            item = _mapping(step, "dish.cooking_steps[]")
            steps.append(CookingStep(
                order=int(item["order"]),
                method=_enum(CookingMethod, item["method"], "cooking_steps.method").value,
                method_name=_string(item["method_name"], "cooking_steps.method_name", required=True),
                ingredient_refs=[_string(ref, "cooking_steps.ingredient_refs[]", required=True) for ref in _list(item.get("ingredient_refs"), "cooking_steps.ingredient_refs")],
                note=_string(item.get("note"), "cooking_steps.note"),
                duration_min=item.get("duration_min"),
                heat_level=_enum(HeatLevel, item.get("heat_level"), "cooking_steps.heat_level"),
            ))
        cuisine_value = value.get("cuisine")
        cuisine = None if cuisine_value is None else CuisineRef(
            name=_string(_mapping(cuisine_value, "dish.cuisine")["name"], "dish.cuisine.name", required=True),
            confidence=float(_mapping(cuisine_value, "dish.cuisine")["confidence"]),
            is_primary=bool(_mapping(cuisine_value, "dish.cuisine").get("is_primary", True)),
        )
        return Dish(
            name=_string(value.get("name"), "dish.name", required=True),
            aliases=[_string(alias, "dish.aliases[]", required=True) for alias in _list(value.get("aliases"), "dish.aliases")],
            dish_type=_enum(DishType, value.get("dish_type"), "dish.dish_type"),
            taste_profile=[_enum(TasteProfile, tag, "dish.taste_profile[]").value for tag in _list(value.get("taste_profile"), "dish.taste_profile")],
            dietary=[_enum(DietaryTag, tag, "dish.dietary[]").value for tag in _list(value.get("dietary"), "dish.dietary")],
            cooking_time_min=value.get("cooking_time_min"), prep_time_min=value.get("prep_time_min"), total_time_min=value.get("total_time_min"),
            difficulty=_enum(Difficulty, value.get("difficulty"), "dish.difficulty"), servings=value.get("servings"),
            calories_per_serving=value.get("calories_per_serving"), description=_string(value.get("description"), "dish.description"),
            cooking_steps=steps, cuisine=cuisine,
        )

    def _ingredient(self, value: dict[str, Any]) -> Ingredient:
        normalized = value.get("amount_normalized")
        amount_normalized = None if normalized is None else AmountNormalized(**_mapping(normalized, "ingredient.amount_normalized"))
        return Ingredient(
            name=_string(value.get("name"), "ingredient.name", required=True),
            category=_enum(IngredientCategory, value.get("category"), "ingredient.category"),
            amount=_string(value.get("amount"), "ingredient.amount", required=True), amount_normalized=amount_normalized,
            is_essential=value.get("is_essential"), preparation=_string(value.get("preparation"), "ingredient.preparation"), notes=_string(value.get("notes"), "ingredient.notes"),
        )

    def _dish_relation(self, value: dict[str, Any]) -> DishRelation:
        relation = _string(value.get("relation"), "dish_relations.relation", required=True)
        if relation not in {"variant_of", "pairs_with"}:
            raise OutputValidationError(f"dish_relations.relation: invalid value {relation!r}")
        return DishRelation(
            from_dish=_string(value.get("from_dish"), "dish_relations.from_dish", required=True),
            to_dish=_string(value.get("to_dish"), "dish_relations.to_dish", required=True), relation=relation,
            variant_type=_enum(VariantType, value.get("variant_type"), "dish_relations.variant_type"),
            context=_string(value.get("context"), "dish_relations.context"), note=_string(value.get("note"), "dish_relations.note"),
        )

    def _ingredient_relation(self, value: dict[str, Any]) -> IngredientRelation:
        relation = _string(value.get("relation"), "ingredient_relations.relation", required=True)
        if relation not in {"complements", "substitutes", "makes"}:
            raise OutputValidationError(f"ingredient_relations.relation: invalid value {relation!r}")
        return IngredientRelation(
            from_ingredient=_string(value.get("from_ingredient"), "ingredient_relations.from_ingredient", required=True),
            to_ingredient=_string(value.get("to_ingredient"), "ingredient_relations.to_ingredient", required=True), relation=relation,
            strength=_enum(ComplementStrength, value.get("strength"), "ingredient_relations.strength"),
            context=_string(value.get("context"), "ingredient_relations.context"),
            direction=_enum(SubstituteDirection, value.get("direction"), "ingredient_relations.direction"),
            impact=_enum(SubstituteImpact, value.get("impact"), "ingredient_relations.impact"),
            condition=_string(value.get("condition"), "ingredient_relations.condition"), process=_string(value.get("process"), "ingredient_relations.process"),
            is_reversible=bool(value.get("is_reversible", False)), note=_string(value.get("note"), "ingredient_relations.note"),
        )
