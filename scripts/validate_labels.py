"""Validate bootstrap training.jsonl against the system prompt contract.

Run: python scripts/validate_labels.py data/bootstrap/training.jsonl
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# ── Enums from prompt ──────────────────────────────────────────────────────────
DISH_TYPES = {
    "main_dish", "staple", "soup", "snack", "dessert", "cold_dish",
    "dipping_sauce", "beverage", "baked_goods", "other",
}
TASTE_PROFILES = {
    "spicy", "numbing", "sweet", "sour", "salty", "umami", "bitter",
    "mild", "fragrant", "pungent", "astringent", "oily",
}
DIETARY_TAGS = {
    "vegetarian", "vegan", "halal", "gluten_free", "lactose_free",
    "nut_free", "low_carb", "high_protein", "keto_friendly",
}
INGREDIENT_CATEGORIES = {
    "meat", "poultry", "seafood", "dairy_eggs", "vegetable", "fruit",
    "mushroom_fungi", "soy_products", "grain", "seasoning", "oils_fats",
    "nuts_seeds", "medicinal_herbs", "processed", "other",
}
COOKING_METHODS = {
    "stir-fry", "pan-fry", "deep-fry", "steam", "boil", "roast/bake",
    "braise", "stew", "quick-fry", "flash-fry", "toss/mix",
    "braise-in-sauce", "smoke", "instant-boil", "bake", "simmer",
    "blanch", "slow-cook",
}
HEAT_LEVELS = {"high_heat", "medium_heat", "low_heat", "simmer"}
DIFFICULTIES = {"easy", "medium", "hard"}
VARIANT_TYPES = {
    "ingredient_sub", "regional", "regional_version", "school_version",
    "dietary", "modern",
}
COMPLEMENT_STRENGTHS = {"classic", "common"}
SUBSTITUTE_DIRECTIONS = {"bidirectional", "one_way"}
SUBSTITUTE_IMPACTS = {"minimal", "noticeable", "significant"}


def slug(name: str) -> str:
    return name.lower().replace(" ", "-")


def validate_file(path: Path) -> tuple[list[str], int, int]:
    issues: list[str] = []
    records = []

    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as e:
            issues.append(f"Line {line_no}: invalid JSON: {e}")
            continue
        records.append((line_no, record))

    total = len(records)
    for line_no, record in records:
        item_id = record.get("id", f"line:{line_no}")
        output = record.get("output", {})

        # ── non-recipe ──
        if output.get("dish") is None:
            if record.get("is_not_a_recipe") is not True:
                issues.append(f"[{item_id}] is_not_a_recipe should be True when dish is null")
            if output.get("reason") != "not_a_recipe":
                issues.append(f"[{item_id}] non-recipe output missing reason='not_a_recipe'")
            if output.get("ingredients") != []:
                issues.append(f"[{item_id}] non-recipe ingredients must be []")
            if output.get("ingredient_relations") != []:
                issues.append(f"[{item_id}] non-recipe ingredient_relations must be []")
            continue

        # ── recipe ──
        dish = output.get("dish", {})
        ingredients = output.get("ingredients", [])
        dish_relations = output.get("dish_relations", [])
        ingredient_relations = output.get("ingredient_relations", [])

        if not dish:
            issues.append(f"[{item_id}] recipe has empty dish object")
            continue

        # --- dish.name ---
        if not isinstance(dish.get("name"), str) or not dish["name"].strip():
            issues.append(f"[{item_id}] dish.name missing or empty")

        # --- dish.aliases ---
        if not isinstance(dish.get("aliases"), list):
            issues.append(f"[{item_id}] dish.aliases must be a list")

        # --- dish.dish_type ---
        dt = dish.get("dish_type")
        if dt is not None and dt not in DISH_TYPES:
            issues.append(f"[{item_id}] dish.dish_type invalid: {dt!r}")

        # --- dish.taste_profile ---
        tp = dish.get("taste_profile", [])
        if not isinstance(tp, list):
            issues.append(f"[{item_id}] dish.taste_profile must be a list")
        else:
            for v in tp:
                if v not in TASTE_PROFILES:
                    issues.append(f"[{item_id}] dish.taste_profile invalid value: {v!r}")

        # --- dish.dietary ---
        diet = dish.get("dietary", [])
        if not isinstance(diet, list):
            issues.append(f"[{item_id}] dish.dietary must be a list")
        else:
            for v in diet:
                if v not in DIETARY_TAGS:
                    issues.append(f"[{item_id}] dish.dietary invalid value: {v!r}")

        # --- dish time fields ---
        for tf in ("cooking_time_min", "prep_time_min", "total_time_min"):
            val = dish.get(tf)
            if val is not None and not isinstance(val, (int, float)):
                issues.append(f"[{item_id}] dish.{tf} must be int or null, got {type(val).__name__}")

        ct = dish.get("cooking_time_min") or 0
        pt = dish.get("prep_time_min") or 0
        tt = dish.get("total_time_min")
        if tt is not None and tt < ct + pt:
            issues.append(
                f"[{item_id}] total_time_min({tt}) < cooking_time_min({ct}) + prep_time_min({pt})"
            )

        # --- dish.difficulty ---
        diff = dish.get("difficulty")
        if diff is not None and diff not in DIFFICULTIES:
            issues.append(f"[{item_id}] dish.difficulty invalid: {diff!r}")

        # --- dish.servings ---
        serv = dish.get("servings")
        if serv is not None and not isinstance(serv, (int, float)):
            issues.append(f"[{item_id}] dish.servings must be int or null")

        # --- dish.calories_per_serving ---
        cal = dish.get("calories_per_serving")
        if cal is not None and not isinstance(cal, (int, float)):
            issues.append(f"[{item_id}] dish.calories_per_serving must be int or null")

        # --- dish.description ---
        desc = dish.get("description")
        if desc is not None and (not isinstance(desc, str) or not desc.strip()):
            issues.append(f"[{item_id}] dish.description must be string or null")

        # --- dish.cuisine ---
        cuisine = dish.get("cuisine")
        if cuisine is not None:
            if not isinstance(cuisine, dict):
                issues.append(f"[{item_id}] dish.cuisine must be object or null")
            else:
                if not isinstance(cuisine.get("name"), str) or not cuisine["name"].strip():
                    issues.append(f"[{item_id}] dish.cuisine.name missing or empty")
                conf = cuisine.get("confidence")
                if not isinstance(conf, (int, float)) or not (0 <= conf <= 1):
                    issues.append(f"[{item_id}] dish.cuisine.confidence must be 0-1, got {conf!r}")
                if "is_primary" not in cuisine:
                    issues.append(f"[{item_id}] dish.cuisine.is_primary missing")
                elif not isinstance(cuisine["is_primary"], bool):
                    issues.append(f"[{item_id}] dish.cuisine.is_primary must be bool")

        # --- dish.cooking_steps ---
        steps = dish.get("cooking_steps", [])
        if not isinstance(steps, list) or len(steps) == 0:
            issues.append(f"[{item_id}] dish.cooking_steps must be non-empty list")
        else:
            for i, step in enumerate(steps):
                prefix = f"[{item_id}] cooking_steps[{i}]"
                order = step.get("order")
                if order != i + 1:
                    issues.append(f"{prefix} order={order}, expected {i+1}")
                method = step.get("method")
                if method not in COOKING_METHODS:
                    issues.append(f"{prefix} method invalid: {method!r}")
                mn = step.get("method_name")
                if not isinstance(mn, str) or not mn.strip():
                    issues.append(f"{prefix} method_name missing or empty")
                refs = step.get("ingredient_refs", [])
                if not isinstance(refs, list):
                    issues.append(f"{prefix} ingredient_refs must be a list")
                heat = step.get("heat_level")
                if heat is not None and heat not in HEAT_LEVELS:
                    issues.append(f"{prefix} heat_level invalid: {heat!r}")
                dur = step.get("duration_min")
                if dur is not None and not isinstance(dur, (int, float)):
                    issues.append(f"{prefix} duration_min must be int or null")

        # --- build ingredient slug lookup ---
        ingredient_slugs = {}
        for ing in ingredients:
            name = ing.get("name", "")
            ingredient_slugs[slug(name)] = name

        # --- ingredient validation ---
        if not isinstance(ingredients, list) or len(ingredients) == 0:
            issues.append(f"[{item_id}] ingredients must be non-empty list")
        else:
            for i, ing in enumerate(ingredients):
                prefix = f"[{item_id}] ingredients[{i}]"
                name = ing.get("name")
                if not isinstance(name, str) or not name.strip():
                    issues.append(f"{prefix} name missing or empty")
                cat = ing.get("category")
                if cat not in INGREDIENT_CATEGORIES:
                    issues.append(f"{prefix} category invalid: {cat!r}")
                amt = ing.get("amount")
                if not isinstance(amt, str) or not amt.strip():
                    issues.append(f"{prefix} amount missing or empty")
                norm = ing.get("amount_normalized")
                if norm is not None:
                    if not isinstance(norm, dict):
                        issues.append(f"{prefix} amount_normalized must be object or null")
                    else:
                        if "value" not in norm:
                            issues.append(f"{prefix} amount_normalized.value missing")
                        if "unit" not in norm:
                            issues.append(f"{prefix} amount_normalized.unit missing")
                is_ess = ing.get("is_essential")
                if not isinstance(is_ess, bool):
                    issues.append(f"{prefix} is_essential must be bool, got {type(is_ess).__name__}")

        # --- cross-check ingredient_refs vs ingredient names ---
        for i, step in enumerate(steps):
            refs = step.get("ingredient_refs", [])
            for ref in refs:
                if ref not in ingredient_slugs:
                    issues.append(
                        f"[{item_id}] cooking_steps[{i}].ingredient_refs[{ref!r}] "
                        f"not found in ingredients (available: {sorted(ingredient_slugs)})"
                    )

        # --- dish_relations ---
        primary_slug = slug(dish.get("name", ""))
        for i, dr in enumerate(dish_relations):
            prefix = f"[{item_id}] dish_relations[{i}]"
            rel = dr.get("relation")
            if rel not in ("variant_of", "pairs_with"):
                issues.append(f"{prefix} relation invalid: {rel!r}")
            from_dish = dr.get("from_dish", "")
            if from_dish != primary_slug:
                issues.append(f"{prefix} from_dish={from_dish!r}, expected primary slug {primary_slug!r}")
            to_dish = dr.get("to_dish")
            if not isinstance(to_dish, str) or not to_dish.strip():
                issues.append(f"{prefix} to_dish missing or empty")
            vt = dr.get("variant_type")
            if vt is not None and vt not in VARIANT_TYPES:
                issues.append(f"{prefix} variant_type invalid: {vt!r}")

        # --- ingredient_relations ---
        relation_specific_fields = {
            "complements": {"strength", "context", "note"},
            "substitutes": {"direction", "impact", "condition", "process", "is_reversible", "note", "context"},
            "makes": {"process", "condition", "note", "context"},
        }
        for i, ir_ in enumerate(ingredient_relations):
            prefix = f"[{item_id}] ingredient_relations[{i}]"
            rel = ir_.get("relation")
            if rel not in ("complements", "substitutes", "makes"):
                issues.append(f"{prefix} relation invalid: {rel!r}")
            from_ing = ir_.get("from_ingredient", "")
            if from_ing not in ingredient_slugs:
                issues.append(f"{prefix} from_ingredient={from_ing!r} not in ingredient slugs")
            to_ing = ir_.get("to_ingredient", "")
            if to_ing not in ingredient_slugs:
                issues.append(f"{prefix} to_ingredient={to_ing!r} not in ingredient slugs")
            if rel == "complements":
                strength = ir_.get("strength")
                if strength is not None and strength not in COMPLEMENT_STRENGTHS:
                    issues.append(f"{prefix} strength invalid: {strength!r}")
            if rel == "substitutes":
                direction = ir_.get("direction")
                if direction is not None and direction not in SUBSTITUTE_DIRECTIONS:
                    issues.append(f"{prefix} direction invalid: {direction!r}")
                impact = ir_.get("impact")
                if impact is not None and impact not in SUBSTITUTE_IMPACTS:
                    issues.append(f"{prefix} impact invalid: {impact!r}")

    return issues, total, sum(1 for _, r in records if r.get("output", {}).get("dish") is not None)


def main():
    if len(sys.argv) < 2:
        path = Path("data/bootstrap/training.jsonl")
    else:
        path = Path(sys.argv[1])

    if not path.exists():
        print(f"ERROR: {path} not found")
        sys.exit(1)

    issues, total, recipe_count = validate_file(path)
    non_recipe = total - recipe_count

    print(f"Total records: {total}")
    print(f"  Recipes: {recipe_count}")
    print(f"  Non-recipes: {non_recipe}")
    print()

    if issues:
        print(f"ISSUES FOUND: {len(issues)}")
        print("-" * 60)
        for issue in issues:
            print(f"  ✗ {issue}")
        print("-" * 60)
        print(f"\nFAIL: {len(issues)} issue(s) found.")
        sys.exit(1)
    else:
        print("All records pass validation against the prompt contract.")
        sys.exit(0)


if __name__ == "__main__":
    main()
