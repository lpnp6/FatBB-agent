"""Contract tests between labeling models and the prompt output schema."""

from __future__ import annotations

import unittest

from labeling.models import DishRelation, ExtractionOutput, IngredientRelation


class LabelingModelContractTests(unittest.TestCase):
    def test_extraction_output_keeps_standalone_relation_lists(self) -> None:
        output = ExtractionOutput(
            dish_relations=[DishRelation(
                from_dish="kung-pao-chicken",
                to_dish="kung-pao-shrimp",
                relation="variant_of",
            )],
            ingredient_relations=[IngredientRelation(
                from_ingredient="chicken-breast",
                to_ingredient="chicken-thigh",
                relation="substitutes",
            )],
        )

        self.assertEqual(output.dish_relations[0].from_dish, "kung-pao-chicken")
        self.assertEqual(output.ingredient_relations[0].to_ingredient, "chicken-thigh")
        self.assertFalse(hasattr(output, "has_ingredient_relations"))
