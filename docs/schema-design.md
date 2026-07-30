# Food Knowledge Graph Schema Design

> Version: v1.2
> Date: 2026-07-23
> Target: Define the complete data structure for multi-source food data extraction and knowledge graph construction

---

## 1. Design Decisions Summary

Eight key decisions made before entering the detailed schema:

| # | Topic | Decision | Rationale |
|---|-------|----------|-----------|
| ① | `taste_profile` as property vs node | **Property** | Taste tags are enumerable (~15-20 values); array property is more query-efficient than standalone nodes for "spicy" |
| ② | Ingredient `category` as property vs node | **Property** | Single-value classification, not a many-to-many relationship — property suffices |
| ③ | Cuisine flat vs hierarchical | **Hierarchical** | Supports tree structure: Chinese → Sichuan → Chengdu-school via `parent` field |
| ④ | Cooking method modeling | **`cooking_steps` property array** | Array order = cooking order; each step references ingredients; see Section 4 |
| ⑤ | Dish-Dish "similar" edge | **Skip** | Queryable via cuisine nodes + taste profile properties; avoids N×N edge explosion |
| ⑥ | `amount_normalized` | **Include**, optional | Preserve original text while providing normalized values for quantity-based queries |
| ⑦ | Cuisine belonging confidence | **Include**, float 0-1 | Some dishes cross cuisines (e.g. General Tso's Chicken is both Hunan and American-Chinese) |
| ⑧ | Ingredient-Ingredient relations | **Include** | Ingredients have complement/substitute/derivation relationships; enriches graph semantics |

---

## 2. Graph Structure Overview

```
                         ┌──────────────────┐
                         │     Cuisine      │
                         │  (hierarchical)  │
                         │  parent → parent │
                         └────────┬─────────┘
                                  │ BELONGS_TO {confidence, is_primary}
        ┌─────────────────────────┼─────────────────────┐
        │                         │                     │
        ▼                         ▼                     ▼
   ┌──────────┐            ┌──────────┐           ┌──────────┐
   │   Dish   │  VARIANT_OF│   Dish   │ PAIRS_WITH│   Dish   │
   │Kung Pao  │◄──────────►│Kung Pao  │◄─────────►│  Steamed │
   │ Chicken  │            │ Shrimp   │           │   Rice   │
   │          │            │          │           │          │
   │ cooking_ │            │ cooking_ │           │ cooking_ │
   │  steps[] │            │  steps[] │           │  steps[] │
   └────┬─────┘            └────┬─────┘           └────┬─────┘
        │                       │                      │
        │ CONTAINS              │ CONTAINS             │ CONTAINS
        │ {amount,              │ {amount,             │ {amount,
        │  normalized,          │  normalized,         │  normalized,
        │  is_essential,        │  is_essential,       │  is_essential,
        │  preparation}         │  preparation}        │  preparation}
        │                       │                      │
        ▼                       ▼                      ▼
   ┌──────────┐◄────────►┌──────────┐◄────────►┌──────────┐
   │Ingredient│ COMPLEMENTS│Ingredient│ SUBSTITUTES│Ingredient│
   │ Chicken  │           │  Shrimp  │           │   Rice   │
   │  Breast  │           │          │           │          │
   │ category │           │ category │           │ category │
   │ nutrition│           │ nutrition│           │ nutrition│
   └──────────┘           └────┬─────┘           └──────────┘
                               │
                               │ MAKES
                               ▼
                          ┌──────────┐
                          │Ingredient│
                          │   Salt   │
                          │ category │
                          └──────────┘
```

**Core principles**:
- **Dish** and **Ingredient** are the two core entity nodes; **Cuisine** is an auxiliary classification node
- **Relationships carry information** (amount, confidence, preparation, etc.)
- Taste and dietary tags are enumerable Dish array properties, not standalone nodes
- Ingredients form their own subgraph via complement/substitute/derivation relationships

---

## 3. Node Type Definitions

### 3.1 Dish

```
Dish {
  // ── Identity ──
  id:              string         // slug, "kung-pao-chicken"
  name:            string         // Primary name in English, "Kung Pao Chicken"
  aliases:         string[]       // Multilingual aliases ["宫保鸡丁", "Kung Pao", "Gong Bao Ji Ding"]

  // ── Classification ──
  dish_type:       enum?          // main_dish | staple | soup | snack | dessert |
                                  //   cold_dish | dipping_sauce | beverage | baked_goods | other

  // ── Taste & Diet ──
  taste_profile:   string[]       // ["spicy","sweet","sour","numbing","salty","umami",
                                  //   "bitter","mild","fragrant","pungent","astringent","oily"]
  dietary:         string[]       // ["vegetarian","vegan","halal","gluten_free","lactose_free",
                                  //   "nut_free","low_carb","high_protein","keto_friendly"]

  // ── Time & Difficulty ──
  cooking_time_min: int?          // Active cooking time in minutes
  prep_time_min:    int?          // Prep time in minutes
  total_time_min:   int?          // Total time in minutes
  difficulty:       enum?         // easy | medium | hard

  // ── Yield & Calories ──
  servings:             int?      // Number of servings
  calories_per_serving: int?      // Calories per serving (kcal)

  // ── Cooking Steps (core design, see Section 4) ──
  cooking_steps:      CookingStep[]

  // ── Description ──
  description:    string?         // Short description (1-3 sentences, English)
  tips:           string[]?       // Cooking tips

  // ── Metadata ──
  source_urls:    string[]        // Source URL list
  extracted_at:   datetime        // Extraction timestamp
  confidence:     float?          // Overall extraction confidence (0-1)
}
```

### 3.2 Ingredient

```
Ingredient {
  // ── Identity ──
  id:              string         // slug, "chicken-breast"
  name:            string         // "Chicken Breast"
  aliases:         string[]       // ["鸡胸肉", "鸡胸", "chicken breast fillet"]

  // ── Classification (property, not node) ──
  category:        enum           // meat | poultry | seafood | dairy_eggs | vegetable |
                                  //   fruit | mushroom_fungi | soy_products | grain |
                                  //   seasoning | oils_fats | nuts_seeds | medicinal_herbs |
                                  //   processed | other
  sub_category:    string?        // Finer classification: "leafy_green" "root" "gourd" "shellfish" etc.

  // ── Nutrition (all optional) ──
  nutrition_per_100g: {
    calories:      float?
    protein_g:     float?
    fat_g:         float?
    carbs_g:       float?
    fiber_g:       float?
    sodium_mg:     float?
  }

  // ── Characteristics ──
  storage:         enum?          // room_temp | refrigerated | frozen
  umami_level:     enum?          // high | medium | low
  season:          string[]?      // Seasonal availability ["spring","summer","autumn","winter"]

  // ── Description ──
  description:     string?        // English description

  // ── Metadata ──
  source_urls:     string[]
}
```

### 3.3 Cuisine (Hierarchical)

```
Cuisine {
  // ── Identity ──
  id:              string         // slug, "sichuan-cuisine"
  name:            string         // "Sichuan Cuisine"
  aliases:         string[]       // ["川菜", "Sichuan food", "Szechuan cuisine"]

  // ── Hierarchy ──
  parent:          string?        // Parent cuisine id, null = top-level
                                  //   Chinese Cuisine → Sichuan Cuisine → Chengdu Cuisine

  // ── Geography ──
  region:          string?        // "Sichuan Province"
  country:         string?        // "China"

  // ── Characteristics ──
  characteristics: string[]       // ["numbing-spicy","fish-fragrant","strange-flavor","oil-heavy"]
  description:     string?        // English description
}
```

**Cuisine hierarchy example**:
```
Chinese Cuisine (parent: null)
  ├── Sichuan Cuisine (parent: "chinese-cuisine")
  │   ├── Chengdu Cuisine (parent: "sichuan-cuisine")
  │   └── Chongqing Cuisine (parent: "sichuan-cuisine")
  ├── Cantonese Cuisine (parent: "chinese-cuisine")
  ├── Shandong Cuisine (parent: "chinese-cuisine")
  ├── Jiangsu Cuisine (parent: "chinese-cuisine")
  └── ...
Japanese Cuisine (parent: null)
  ├── Kaiseki (parent: "japanese-cuisine")
  └── ...
```

---

## 4. cooking_steps Structure (Key Design)

### 4.1 Structure Definition

```typescript
CookingStep {
  order:           int            // Step number, starting from 1; order = cooking sequence
  method:          string         // Cooking method slug, e.g. "stir-fry" "deep-fry" "steam"
  method_name:     string         // Human-readable method name, e.g. "Stir-Fry" "Deep-Fry"
  ingredient_refs: string[]       // Ingredient IDs involved in this step
  note:            string?        // Instruction — "Fry peanuts until golden and crispy"
  duration_min:    int?           // Step duration in minutes
  heat_level:      enum?          // Heat level: high_heat | medium_heat | low_heat | simmer
}
```

### 4.2 Example: Kung Pao Chicken

```json
{
  "cooking_steps": [
    {
      "order": 1,
      "method": "deep-fry",
      "method_name": "Deep-Fry",
      "ingredient_refs": ["peanuts"],
      "note": "Fry peanuts until golden and crispy, drain and set aside",
      "duration_min": 2,
      "heat_level": "medium_heat"
    },
    {
      "order": 2,
      "method": "stir-fry",
      "method_name": "Stir-Fry",
      "ingredient_refs": ["chicken-breast"],
      "note": "Velvet chicken cubes in hot oil until just white, remove and set aside",
      "duration_min": 3,
      "heat_level": "high_heat"
    },
    {
      "order": 3,
      "method": "stir-fry",
      "method_name": "Stir-Fry",
      "ingredient_refs": ["dried-chili", "sichuan-peppercorn", "scallion", "ginger", "garlic"],
      "note": "Keep some oil in wok, bloom chilies, peppercorns, and aromatics",
      "duration_min": 1,
      "heat_level": "high_heat"
    },
    {
      "order": 4,
      "method": "stir-fry",
      "method_name": "Stir-Fry",
      "ingredient_refs": ["chicken-breast", "peanuts", "kung-pao-sauce"],
      "note": "Return chicken and peanuts to wok, add Kung Pao sauce, toss until glazed",
      "duration_min": 2,
      "heat_level": "high_heat"
    }
  ]
}
```

### 4.3 Why as a Dish Property Instead of Standalone Nodes?

| Consideration | Property Approach ✅ | Standalone Node Approach ❌ |
|---------------|---------------------|----------------------------|
| Query complexity | `MATCH (d) WHERE d.cooking_steps[0].method = 'deep-fry'` — simple | Requires multi-hop traversal of Step nodes |
| LLM extraction | Direct nested JSON output; no post-processing needed | Requires splitting into nodes and relationships |
| Data consistency | Steps are intrinsic to the dish; no independent identity | Over-normalized: introduces extra CookingStep nodes |
| Cross-dish sharing | Not needed (each dish's process is unique) | Over-engineering |

### 4.4 CookingMethod as a Reference Table

While `cooking_steps` is a Dish property, cooking method names still need a **reference table** for consistency. This is used at the extraction layer and optionally stored as lightweight nodes in the graph:

```
CookingMethod (reference table / optional lightweight node) {
  id:              string         // "stir-fry"
  name:            string         // "Stir-Fry"
  aliases:         string[]       // ["炒", "翻炒", "爆炒", "wok-fry"]
  category:        enum?          // dry_heat | moist_heat | oil_medium | no_heat | other
  description:     string?
}
```

Dish.`cooking_steps[].method` references `CookingMethod.id`. If the LLM encounters a method not in the reference table, it should map to the closest existing method or flag it for review.

---

## 5. Relationship Type Definitions

### 5.1 CONTAINS (Dish → Ingredient)

The core relationship. A Dish contains an Ingredient.

```
Dish -[CONTAINS]→ Ingredient
```

| Property | Type | Required | Description |
|----------|------|----------|-------------|
| `amount` | string | Yes | Original quantity text — "200g" "2 eggs" "to taste" "a pinch" |
| `amount_normalized` | object? | No | Normalized quantity, see structure below |
| `is_essential` | bool | Yes | true = main ingredient, false = auxiliary/seasoning/garnish |
| `preparation` | string? | No | Pre-processing — "julienned" "minced" "pre-soaked" |
| `notes` | string? | No | Additional notes — "can substitute with XX" "optional" |

**amount_normalized structure**:
```json
{
  "value": 200,
  "unit": "g",
  "range_low": null,
  "range_high": null
}
```

- Supports exact values (`value: 200, unit: "g"`)
- Supports ranges (`range_low: 200, range_high: 300, unit: "g"`)
- For unquantifiable expressions like "a pinch" or "to taste", this field is `null`

### 5.2 BELONGS_TO (Dish → Cuisine)

```
Dish -[BELONGS_TO]→ Cuisine
```

| Property | Type | Required | Description |
|----------|------|----------|-------------|
| `confidence` | float | Yes | Belonging confidence (0-1) |
| `is_primary` | bool | No | Whether this is the primary cuisine affiliation; default true |

- A dish can belong to multiple cuisines (e.g. "General Tso's Chicken" links to both Hunan and American-Chinese)
- `confidence` reflects how certain the affiliation is
- `is_primary` marks the most central affiliation

### 5.3 VARIANT_OF (Dish ↔ Dish)

```
Dish -[VARIANT_OF]→ Dish
```

| Property | Type | Required | Description |
|----------|------|----------|-------------|
| `variant_type` | enum | Yes | ingredient_sub / regional / regional_version / school_version / dietary / modern |
| `note` | string? | No | Description of the variant difference |

Example:
```
Kung Pao Shrimp -[VARIANT_OF {type:"ingredient_sub"}]→ Kung Pao Chicken
```

### 5.4 PAIRS_WITH (Dish ↔ Dish)

```
Dish -[PAIRS_WITH]→ Dish
```

| Property | Type | Required | Description |
|----------|------|----------|-------------|
| `context` | string? | No | Pairing context — "breakfast" "with_drinks" "traditional_pairing" |

Example:
```
Soy Milk -[PAIRS_WITH {context:"breakfast"}]→ Youtiao (Chinese Cruller)
```

### 5.5 Ingredient-to-Ingredient Relations

Added in v1.1. Upgrades ingredients from "isolated leaf nodes" to an "interconnected subgraph" — a key semantic enrichment.

#### 5.5.1 COMPLEMENTS (Perfect Match)

Two ingredients that work far better together than separately. The most important ingredient relationship.

```
Ingredient -[COMPLEMENTS]→ Ingredient
```

| Property | Type | Required | Description |
|----------|------|----------|-------------|
| `strength` | enum? | No | classic (iconic pair) / common (frequent pairing) |
| `context` | string? | No | Usage context — "removes_gaminess" "enhances_umami" "adds_aroma" |
| `note` | string? | No | Explanation of the complementarity |

Examples:
```
Sichuan Peppercorn -[COMPLEMENTS {strength:"classic", context:"adds_numbing_and_aroma"}]→ Dried Chili
Tomato -[COMPLEMENTS {strength:"classic"}]→ Egg
Garlic -[COMPLEMENTS {strength:"common", context:"aroma_and_deodorize"}]→ Water Spinach
```

#### 5.5.2 SUBSTITUTES (Replacement)

One ingredient can be replaced by another without significantly altering the dish.

```
Ingredient -[SUBSTITUTES]→ Ingredient
```

| Property | Type | Required | Description |
|----------|------|----------|-------------|
| `direction` | enum? | No | bidirectional (two-way) / one_way (one-way substitution) |
| `impact` | enum? | No | minimal (barely detectable) / noticeable (detectable but not jarring) / significant (markedly different) |
| `condition` | string? | No | Constraint — "similar_texture" "same_poultry_family" "vegetarian_substitute" |
| `note` | string? | No | Description |

Examples:
```
Chicken Breast -[SUBSTITUTES {direction:"bidirectional", impact:"minimal", condition:"similar_texture"}]→ Chicken Thigh
Butter -[SUBSTITUTES {direction:"bidirectional", impact:"noticeable", condition:"interchangeable_in_baking"}]→ Lard
Tofu -[SUBSTITUTES {direction:"one_way", impact:"significant", condition:"vegetarian_meat_substitute"}]→ Pork
```

#### 5.5.3 MAKES (Processing Chain)

One ingredient is produced by processing another ingredient.

```
Ingredient -[MAKES]→ Ingredient
```

| Property | Type | Required | Description |
|----------|------|----------|-------------|
| `process` | string? | No | Processing method — "fermentation" "pickling" "grinding" "pressing" "coagulation" |
| `is_reversible` | bool | No | Whether the process is reversible; default false |
| `note` | string? | No | Description |

Examples:
```
Soybean -[MAKES {process:"soaking→grinding→boiling→coagulation→molding"}]→ Tofu
Tofu -[MAKES {process:"fermentation"}]→ Fermented Tofu
Milk -[MAKES {process:"fermentation"}]→ Yogurt
Pork Belly -[MAKES {process:"pickling→air_drying"}]→ Ham
```

---

## 6. Relationship Summary

| Relationship | Source | Target | Direction | Key Properties |
|-------------|--------|--------|-----------|----------------|
| `CONTAINS` | Dish | Ingredient | → | amount, amount_normalized, is_essential, preparation |
| `BELONGS_TO` | Dish | Cuisine | → | confidence, is_primary |
| `VARIANT_OF` | Dish | Dish | ↔ | variant_type, note |
| `PAIRS_WITH` | Dish | Dish | ↔ | context |
| `COMPLEMENTS` | Ingredient | Ingredient | ↔ | strength, context, note |
| `SUBSTITUTES` | Ingredient | Ingredient | ↔ | direction, impact, condition, note |
| `MAKES` | Ingredient | Ingredient | → | process, is_reversible, note |

**Excluded relationships and their alternatives**:
- ~~SIMILAR_TASTE~~ → Query via `WHERE d.taste_profile && ['spicy','numbing']` property matching
- ~~SAME_CUISINE~~ → Already handled by `BELONGS_TO`; dishes are indirectly related through cuisine nodes
- ~~SHARED_INGREDIENT~~ → Query by traversing CONTAINS relationships

**Ingredient relationship boundaries**:
- `COMPLEMENTS` is limited to combinations where pairing is **significantly better than using either alone**, not "they once appeared in the same dish"
- `SUBSTITUTES` is about **flavor/texture** substitutability, not nutritional equivalence
- `MAKES` records **physical/chemical processing** relationships, not logical classification (use `category` for that)
- `CONTAINS` already handles "what ingredients are in this dish" and "which dishes use this ingredient"; ingredient-ingredient relations only cover intrinsic ingredient-to-ingredient connections

---

## 7. Multilingual Design

### 7.1 Design Philosophy

The knowledge graph stores all structured data in **English** while preserving original-language information through aliases. The fine-tuned model handles multilingual input (Chinese, English, Japanese, etc.) and normalizes output to English.

### 7.2 Model Input

The model accepts food-related Markdown text in **any language**:

- Chinese recipe pages (下厨房, 美食天下)
- English Wikipedia articles
- Japanese recipe sites (Cookpad)
- Korean food blogs
- Mixed-language content (e.g. Chinese text with English loanwords)

### 7.3 Model Output

Always English-structured JSON. The model performs **extraction + translation** in a single pass:

| Text in source language | Output field | Output value |
|--------------------------|-------------|--------------|
| 宫保鸡丁 | `name` | "Kung Pao Chicken" |
| 宫保鸡丁 / 宫保鸡 / Kung Pao Chicken | `aliases` | ["宫保鸡丁", "宫保鸡", "Kung Pao Chicken"] |
| 鸡胸肉切丁 | `preparation` | "Diced into cubes" |
| 炒至金黄酥脆 | `note` | "Stir-fry until golden and crispy" |

### 7.4 Multilingual Aliases

The `aliases` field on every node serves as the cross-lingual bridge. It collects names from all source languages:

```
Dish {
  name: "Kung Pao Chicken",          // Canonical English name
  aliases: [                          // All names encountered across languages
    "宫保鸡丁",                        // Chinese (Simplified)
    "宮保雞丁",                        // Chinese (Traditional)
    "宫保鸡",                          // Chinese short name
    "Kung Pao Chicken",               // English
    "Gong Bao Ji Ding"                // Pinyin romanization
  ]
}
```

This enables:
- **Search**: find "Kung Pao Chicken" whether the user types English, Chinese, or Pinyin
- **Dedup**: merge dishes scraped from different language sources into one node
- **Preservation**: original names are never lost

### 7.5 Enum Values Are Always English

All enum-valued properties use English slugs:

```
taste_profile:  ["spicy", "sweet", "sour"]     NOT ["辣", "甜", "酸"]
dish_type:      "main_dish"                    NOT "主菜"
category:       "poultry"                      NOT "禽类"
```

This keeps queries language-agnostic: `WHERE 'spicy' IN d.taste_profile` works regardless of the source language.

### 7.6 Free-Text Fields

Description, notes, tips, and other free-text fields are stored in **English**. The model is trained to translate these during extraction. Quality expectation: the 3B model can produce fluent translations for food-domain text (a relatively narrow, high-resource domain).

---

## 8. LLM Extraction Output Format

The fine-tuned model takes a food page (Markdown, any language) and outputs the following JSON structure:

```json
{
  "dish": {
    "name": "Kung Pao Chicken",
    "aliases": ["宫保鸡丁", "宫保鸡", "Kung Pao Chicken", "Gong Bao Ji Ding"],
    "dish_type": "main_dish",
    "taste_profile": ["spicy", "sweet", "sour", "numbing"],
    "dietary": ["nut_free"],
    "cooking_time_min": 15,
    "prep_time_min": 20,
    "total_time_min": 35,
    "difficulty": "medium",
    "servings": 4,
    "calories_per_serving": 380,
    "description": "Kung Pao Chicken is a world-famous classic Sichuan dish featuring diced chicken stir-fried with peanuts, dried chilies, and Sichuan peppercorns in a sweet-savory-spicy sauce.",
    "cooking_steps": [
      {
        "order": 1,
        "method": "deep-fry",
        "method_name": "Deep-Fry",
        "ingredient_refs": ["peanuts"],
        "note": "Fry peanuts until golden and crispy, drain and set aside",
        "duration_min": 2,
        "heat_level": "medium_heat"
      }
    ],
    "cuisine": {
      "name": "Sichuan Cuisine",
      "confidence": 0.95,
      "is_primary": true
    }
  },
  "ingredients": [
    {
      "name": "Chicken Breast",
      "category": "poultry",
      "amount": "300g",
      "amount_normalized": {"value": 300, "unit": "g"},
      "is_essential": true,
      "preparation": "Diced into 1.5cm cubes"
    },
    {
      "name": "Peanuts",
      "category": "nuts_seeds",
      "amount": "50g",
      "amount_normalized": {"value": 50, "unit": "g"},
      "is_essential": true,
      "preparation": "Deep-fried until crispy"
    },
    {
      "name": "Dried Chili",
      "category": "seasoning",
      "amount": "10 pieces",
      "amount_normalized": null,
      "is_essential": true,
      "preparation": "Cut into sections"
    }
  ],
  "dish_relations": [
    {
      "from_dish": "kung-pao-chicken",
      "to_dish": "kung-pao-shrimp",
      "relation": "variant_of",
      "variant_type": "ingredient_sub",
      "context": null,
      "note": null
    }
  ],
  "ingredient_relations": [
    {
      "from_ingredient": "sichuan-peppercorn",
      "to_ingredient": "dried-chili",
      "relation": "complements",
      "strength": "classic",
      "context": "adds_numbing_and_aroma",
      "note": "The numbing of Sichuan peppercorn and dried chili form the core of Sichuan mala flavor."
    }
  ]
}
```

**Field mapping**: LLM output → Graph structure

| LLM Output Path | Graph Target |
|-----------------|--------------|
| `dish.*` (excluding cuisine/ingredients) | Dish node properties |
| `dish.cuisine` + `dish.name` | `(Dish)-[:BELONGS_TO {confidence}]->(Cuisine)` |
| `ingredients[]` | `(Ingredient)` nodes + `(Dish)-[:CONTAINS {amount,...}]->(Ingredient)` |
| `dish_relations[]` | `(Dish)-[:VARIANT_OF {variant_type, note}]->(Dish)` or `(Dish)-[:PAIRS_WITH {context}]->(Dish)` |
| `dish.cooking_steps[]` | `cooking_steps` property on Dish node |
| `dish.cooking_steps[].ingredient_refs[]` | References Ingredient by id, linked to `ingredients[].name` |
| `ingredient_relations[]` | `(Ingredient)-[:COMPLEMENTS | SUBSTITUTES | MAKES]->(Ingredient)` |

**ingredient_relations format** (extracted independently, not nested inside dish):

```json
{
  "ingredient_relations": [
    {
      "from_ingredient": "sichuan-peppercorn",
      "to_ingredient": "dried-chili",
      "relation": "complements",
      "strength": "classic",
      "context": "adds_numbing_and_aroma",
      "note": "The numbing of Sichuan peppercorn and the heat of dried chili form the core of Sichuan mala flavor profile"
    },
    {
      "from_ingredient": "chicken-breast",
      "to_ingredient": "chicken-thigh",
      "relation": "substitutes",
      "direction": "bidirectional",
      "impact": "minimal",
      "condition": "similar_texture_same_poultry_family"
    },
    {
      "from_ingredient": "soybean",
      "to_ingredient": "tofu",
      "relation": "makes",
      "process": "soaking→grinding→boiling→coagulation→molding",
      "is_reversible": false
    }
  ]
}
```

> **Note**: ingredient_relations and Dish extraction are **independent**. Ingredient relations can be extracted from recipe pages (e.g. "Sichuan peppercorn pairs with chili") or from ingredient-specific Wiki pages. The LLM outputs ingredient relations whenever it encounters descriptions of inter-ingredient relationships.

---

## 9. Representative Queries (Cypher)

### Q1: What ingredients does a dish need?
```cypher
MATCH (d:Dish {id: "kung-pao-chicken"})-[r:CONTAINS]->(i:Ingredient)
RETURN i.name, r.amount, r.is_essential, r.preparation
ORDER BY r.is_essential DESC
```

### Q2: What dishes can I make with chicken breast?
```cypher
MATCH (d:Dish)-[r:CONTAINS]->(i:Ingredient {id: "chicken-breast"})
WHERE r.is_essential = true
RETURN d.name, d.taste_profile
```

### Q3: All dishes in Sichuan cuisine (including sub-cuisines)?
```cypher
MATCH (c:Cuisine {id: "sichuan-cuisine"})
MATCH (sub:Cuisine) WHERE sub.parent = c.id OR sub.id = c.id
MATCH (d:Dish)-[:BELONGS_TO]->(sub)
RETURN DISTINCT d.name, sub.name AS cuisine
```

### Q4: Dishes with the closest taste profile to Kung Pao Chicken?
```cypher
MATCH (d:Dish {id: "kung-pao-chicken"})
WITH d.taste_profile AS target_tastes
MATCH (other:Dish)
WHERE other.id <> "kung-pao-chicken"
  AND any(t IN other.taste_profile WHERE t IN target_tastes)
WITH other, target_tastes,
     size([t IN other.taste_profile WHERE t IN target_tastes]) AS overlap
RETURN other.name, other.taste_profile, overlap
ORDER BY overlap DESC LIMIT 10
```

### Q5: Sichuan dishes that use Sichuan peppercorn?
```cypher
MATCH (d:Dish)-[:BELONGS_TO]->(c:Cuisine {id: "sichuan-cuisine"})
MATCH (d)-[:CONTAINS]->(i:Ingredient {id: "sichuan-peppercorn"})
RETURN d.name
```

### Q6: Dishes whose cooking steps include deep-frying?
```cypher
MATCH (d:Dish)
WHERE any(step IN d.cooking_steps WHERE step.method = 'deep-fry')
RETURN d.name, d.cooking_steps
```

### Q7: What variants does a dish have?
```cypher
MATCH (d:Dish {id: "kung-pao-chicken"})-[r:VARIANT_OF]-(v:Dish)
RETURN d.name, v.name, r.variant_type, r.note
```

### Q8: What are substitutes for chicken?
```cypher
MATCH (i:Ingredient {id: "chicken-breast"})-[r:SUBSTITUTES]-(alt:Ingredient)
RETURN i.name, alt.name, r.impact, r.condition
```

### Q9: What are the classic pairings for Sichuan peppercorn?
```cypher
MATCH (i:Ingredient {id: "sichuan-peppercorn"})-[r:COMPLEMENTS]-(other:Ingredient)
RETURN i.name, r.strength, other.name, r.context
ORDER BY r.strength
```

### Q10: What can soybeans be processed into? (including indirect)
```cypher
MATCH path = (i:Ingredient {id: "soybean"})-[:MAKES*1..3]->(derived:Ingredient)
RETURN [node IN nodes(path) | node.name] AS chain,
       [rel IN relationships(path) | rel.process] AS processes
```

### Q11: Substitute ingredients for chicken that also appear in Kung Pao Chicken variants?
```cypher
MATCH (i:Ingredient {id: "chicken-breast"})-[:SUBSTITUTES]-(alt:Ingredient)
MATCH (d:Dish)-[:CONTAINS]->(alt)
MATCH (d)-[:CONTAINS]->(peanut:Ingredient {id: "peanuts"})
RETURN d.name, alt.name
```

---

## 10. Enum Value Definitions

### 10.1 Dish Type (dish_type)
```
main_dish | staple | soup | snack | dessert | cold_dish | dipping_sauce | beverage | baked_goods | other
```

### 10.2 Taste Profile (taste_profile)
```
spicy | numbing | sweet | sour | salty | umami | bitter | mild | fragrant | pungent | astringent | oily
```

### 10.3 Dietary Tags (dietary)
```
vegetarian | vegan | halal | gluten_free | lactose_free | nut_free | low_carb | high_protein | keto_friendly
```

### 10.4 Ingredient Category (Ingredient.category)
```
meat | poultry | seafood | dairy_eggs | vegetable | fruit | mushroom_fungi | soy_products | grain | seasoning | oils_fats | nuts_seeds | medicinal_herbs | processed | other
```

### 10.5 Variant Type (VARIANT_OF.variant_type)
```
ingredient_sub      — Ingredient substitution (Kung Pao Chicken → Kung Pao Shrimp)
regional            — Regional variant (Lanzhou Beef Noodles → Japanese Ramen)
regional_version    — Same dish, different region (Sichuan Kung Pao → American Kung Pao)
school_version      — Same dish, different school (Chengdu Twice-Cooked Pork → Chongqing Twice-Cooked Pork)
dietary             — Dietary variant (Braised Pork Belly → Vegan Braised "Pork")
modern              — Modern reinterpretation (Traditional Mooncake → Snow-Skin Mooncake)
```

### 10.6 Cooking Methods (method — reference table)
```
stir-fry             | pan-fry           | deep-fry
steam                | boil              | roast/bake
braise               | stew              | quick-fry
flash-fry            | toss/mix          | braise-in-sauce
smoke                | instant-boil      | bake
simmer               | blanch            | slow-cook
```

### 10.7 Ingredient Relation Enums

**COMPLEMENTS.strength**:
```
classic   — Iconic, irreplaceable pairing (Sichuan peppercorn + dried chili, tomato + egg, cumin + lamb)
common    — Frequent pairing (garlic + leafy greens, ginger + seafood)
```

**SUBSTITUTES.direction**:
```
bidirectional   — Mutually substitutable (chicken breast ↔ chicken thigh)
one_way         — One-way substitution (tofu → pork as vegetarian substitute; reverse not valid)
```

**SUBSTITUTES.impact**:
```
minimal      — Barely detectable (within the same category)
noticeable   — Detectable but not jarring (butter → lard)
significant  — Markedly different (tofu replacing meat)
```

**MAKES.process** (common values, not a closed enum):
```
fermentation | pickling | air_drying | smoking | grinding | pressing | coagulation | distillation | simmering_reduction | soaking | sprouting
```

---

## 11. Data Pipeline

```
┌──────────────┐     ┌────────────────┐     ┌─────────────────┐
│  Wiki / Recipe│────▶│  trafilatura   │────▶│  Markdown Text  │
│  Multilingual │     │  Content Extr. │     │  (any language) │
│  HTML Sources │     │                │     │                 │
└──────────────┘     └────────────────┘     └───────┬─────────┘
                                                    │
                                                    ▼
                                           ┌─────────────────┐
                                           │ 3B Fine-Tuned   │
                                           │ Model (QLoRA)   │
                                           │                 │
                                           │ Input: Markdown │
                                           │ Output: JSON (EN)│
                                           └───────┬─────────┘
                                                   │
                                                   ▼
┌──────────────┐     ┌────────────────┐     ┌─────────────────┐
│   Neo4j      │◀────│  Mapping Layer │◀────│  JSON Output    │
│   Knowledge  │     │  JSON→Nodes+   │     │  (Section 8)    │
│   Graph      │     │  Relationships │     │                 │
└──────────────┘     └────────────────┘     └─────────────────┘
```

---

## 12. Future Extensions (Reserved)

The following fields/capabilities are not in v1 but are accommodated by the schema:

- **Ingredient nutrition data**: `nutrition_per_100g` field defined; initially mostly empty
- **Seasonal information**: `Ingredient.season`; awaiting seasonal data source integration
- **Cuisine multi-level hierarchy**: `parent` field supports arbitrary depth; initially 1-2 levels
- **Step-level details**: `cooking_steps[].duration_min` and `heat_level` are optional
- **Dish mutual exclusion**: e.g. "vegetarian vs. non-vegetarian version," expressible via VARIANT_OF + variant_type=dietary
- **Multi-serving scaling**: current `servings` + `amount` represents single-batch quantities; multi-serving quantity derivation not yet implemented
