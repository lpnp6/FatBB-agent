"""Enum definitions for the food knowledge graph schema.

All enum values are English slugs per schema-design.md §10.
Source of truth: docs/schema-design.md v1.2.
"""

from __future__ import annotations

from enum import Enum


class DishType(str, Enum):
    MAIN_DISH = "main_dish"
    STAPLE = "staple"
    SOUP = "soup"
    SNACK = "snack"
    DESSERT = "dessert"
    COLD_DISH = "cold_dish"
    DIPPING_SAUCE = "dipping_sauce"
    BEVERAGE = "beverage"
    BAKED_GOODS = "baked_goods"
    OTHER = "other"


class TasteProfile(str, Enum):
    SPICY = "spicy"
    NUMBING = "numbing"
    SWEET = "sweet"
    SOUR = "sour"
    SALTY = "salty"
    UMAMI = "umami"
    BITTER = "bitter"
    MILD = "mild"
    FRAGRANT = "fragrant"
    PUNGENT = "pungent"
    ASTRINGENT = "astringent"
    OILY = "oily"


class DietaryTag(str, Enum):
    VEGETARIAN = "vegetarian"
    VEGAN = "vegan"
    HALAL = "halal"
    GLUTEN_FREE = "gluten_free"
    LACTOSE_FREE = "lactose_free"
    NUT_FREE = "nut_free"
    LOW_CARB = "low_carb"
    HIGH_PROTEIN = "high_protein"
    KETO_FRIENDLY = "keto_friendly"


class IngredientCategory(str, Enum):
    MEAT = "meat"
    POULTRY = "poultry"
    SEAFOOD = "seafood"
    DAIRY_EGGS = "dairy_eggs"
    VEGETABLE = "vegetable"
    FRUIT = "fruit"
    MUSHROOM_FUNGI = "mushroom_fungi"
    SOY_PRODUCTS = "soy_products"
    GRAIN = "grain"
    SEASONING = "seasoning"
    OILS_FATS = "oils_fats"
    NUTS_SEEDS = "nuts_seeds"
    MEDICINAL_HERBS = "medicinal_herbs"
    PROCESSED = "processed"
    OTHER = "other"


class VariantType(str, Enum):
    INGREDIENT_SUB = "ingredient_sub"
    REGIONAL = "regional"
    REGIONAL_VERSION = "regional_version"
    SCHOOL_VERSION = "school_version"
    DIETARY = "dietary"
    MODERN = "modern"


class CookingMethod(str, Enum):
    STIR_FRY = "stir-fry"
    PAN_FRY = "pan-fry"
    DEEP_FRY = "deep-fry"
    STEAM = "steam"
    BOIL = "boil"
    ROAST_BAKE = "roast/bake"
    BRAISE = "braise"
    STEW = "stew"
    QUICK_FRY = "quick-fry"
    FLASH_FRY = "flash-fry"
    TOSS_MIX = "toss/mix"
    BRAISE_IN_SAUCE = "braise-in-sauce"
    SMOKE = "smoke"
    INSTANT_BOIL = "instant-boil"
    BAKE = "bake"
    SIMMER = "simmer"
    BLANCH = "blanch"
    SLOW_COOK = "slow-cook"


class HeatLevel(str, Enum):
    HIGH_HEAT = "high_heat"
    MEDIUM_HEAT = "medium_heat"
    LOW_HEAT = "low_heat"
    SIMMER = "simmer"


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class ComplementStrength(str, Enum):
    CLASSIC = "classic"
    COMMON = "common"


class SubstituteDirection(str, Enum):
    BIDIRECTIONAL = "bidirectional"
    ONE_WAY = "one_way"


class SubstituteImpact(str, Enum):
    MINIMAL = "minimal"
    NOTICEABLE = "noticeable"
    SIGNIFICANT = "significant"


class StorageType(str, Enum):
    ROOM_TEMP = "room_temp"
    REFRIGERATED = "refrigerated"
    FROZEN = "frozen"


class UmamiLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
