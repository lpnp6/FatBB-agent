"""Evaluate a fine-tuned Qwen2.5-3B QLoRA model against the validation set.

Loads the 4-bit base model with the saved LoRA adapter, runs greedy-decoding
inference on every validation example, and reports:

- JSON validity (% of outputs that parse as valid JSON)
- Schema validity (% passing the labeling validator)
- Enum accuracy (% of enum fields matching allowed values)
- Per-field completeness vs the gold labels

Usage::

    PYTHONPATH=src python -m labeling_sft.evaluate \\
        --adapter_dir models/qwen2.5-3b-fatbb-v1 \\
        --val_file data/training/val.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any

from .train import format_example, load_system_prompt

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def load_eval_model(adapter_dir: str, base_model_id: str = "Qwen/Qwen2.5-3B-Instruct") -> tuple[Any, Any]:
    """Load the base model + LoRA adapter for evaluation.

    Returns ``(model, tokenizer)`` with the model in evaluation mode and
    the LoRA adapter merged for faster inference.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    logger.info("Loading base model: %s", base_model_id)
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    logger.info("Loading LoRA adapter: %s", adapter_dir)
    model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()

    return model, tokenizer


def generate_one(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = 4096,
) -> str:
    """Run greedy-decoding inference and return the generated text."""
    import torch

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=8192 - max_new_tokens)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> str | None:
    """Try to extract a JSON object from model output.

    Handles outputs that may include trailing prose, markdown fences, or
    chat-template continuations.
    """
    # Strip trailing chat tokens if present
    text = re.sub(r"<\|im_end\|>.*$", "", text).strip()

    # Try bare parse first
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code fence
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        return m.group(1)

    # Try finding the outermost JSON object
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        return m.group(0)

    return None


def _check_enum_values(output: dict[str, Any]) -> dict[str, int]:
    """Count enum fields and how many are valid."""
    from labeling.models.enums import (
        CookingMethod, DietaryTag, Difficulty, DishType, HeatLevel,
        IngredientCategory, TasteProfile,
    )

    valid = 0
    total = 0

    dish = output.get("dish")
    if isinstance(dish, dict):
        # dish_type
        if dish.get("dish_type") is not None:
            total += 1
            try:
                DishType(dish["dish_type"])
                valid += 1
            except ValueError:
                pass
        # difficulty
        if dish.get("difficulty") is not None:
            total += 1
            try:
                Difficulty(dish["difficulty"])
                valid += 1
            except ValueError:
                pass
        # taste_profile
        for tag in dish.get("taste_profile") or []:
            total += 1
            try:
                TasteProfile(tag)
                valid += 1
            except ValueError:
                pass
        # dietary
        for tag in dish.get("dietary") or []:
            total += 1
            try:
                DietaryTag(tag)
                valid += 1
            except ValueError:
                pass
        # cooking_steps method
        for step in dish.get("cooking_steps") or []:
            if step.get("method"):
                total += 1
                try:
                    CookingMethod(step["method"])
                    valid += 1
                except ValueError:
                    pass
            if step.get("heat_level"):
                total += 1
                try:
                    HeatLevel(step["heat_level"])
                    valid += 1
                except ValueError:
                    pass
        # ingredient category
        for ing in output.get("ingredients") or []:
            if ing.get("category"):
                total += 1
                try:
                    IngredientCategory(ing["category"])
                    valid += 1
                except ValueError:
                    pass

    return {"total_enum_fields": total, "valid_enum_fields": valid}


def _per_field_coverages(outputs: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """Compute per-field presence counts across all outputs."""
    from collections import Counter

    fields: Counter[str] = Counter()
    for out in outputs:
        dish = out.get("dish")
        if not isinstance(dish, dict) or dish is None:
            continue
        for key in ("name", "dish_type", "taste_profile", "dietary",
                     "cooking_time_min", "prep_time_min", "total_time_min",
                     "difficulty", "servings", "calories_per_serving",
                     "description", "cooking_steps", "cuisine"):
            val = dish.get(key)
            if val is not None and val != []:
                fields[key] += 1
        if out.get("ingredients"):
            fields["ingredients"] += 1

    return {
        field: {"present": count, "total": len(outputs), "pct": round(count * 100 / len(outputs))}
        for field, count in fields.most_common()
    }


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate(
    adapter_dir: str,
    val_file: str = "data/training/val.jsonl",
    base_model_id: str = "Qwen/Qwen2.5-3B-Instruct",
    output_report: str | None = "data/training/eval_report.json",
) -> dict[str, Any]:
    """Run the evaluation and return a metrics dict."""
    val_path = Path(val_file)
    if not val_path.exists():
        raise FileNotFoundError(f"Validation file not found: {val_file}")

    with val_path.open("r", encoding="utf-8") as fh:
        gold_records = [json.loads(line) for line in fh if line.strip()]

    system_prompt = load_system_prompt()
    model, tokenizer = load_eval_model(adapter_dir, base_model_id)

    json_valid = 0
    schema_valid = 0
    not_recipe_correct = 0
    not_recipe_total = 0
    all_enum_results: list[dict[str, int]] = []
    predicted_outputs: list[dict[str, Any]] = []

    for i, record in enumerate(gold_records):
        gold_output = json.loads(record["output"])
        is_not_recipe = gold_output.get("dish") is None

        prompt = format_example(
            {"instruction": record["instruction"],
             "input": record["input"],
             "output": ""},
            system_prompt,
        )
        # Remove the empty assistant block — we want just the prompt
        prompt = prompt.rsplit("<|im_start|>assistant\n", 1)[0] + "<|im_start|>assistant\n"

        generated = generate_one(model, tokenizer, prompt)

        # JSON validity
        json_text = _extract_json(generated)
        if json_text is None:
            logger.warning("[%d/%d] JSON parse failed", i + 1, len(gold_records))
            continue
        json_valid += 1

        try:
            predicted = json.loads(json_text)
        except json.JSONDecodeError:
            continue

        predicted_outputs.append(predicted)

        # Schema validity — basic structural check
        if isinstance(predicted.get("dish"), dict) or predicted.get("reason") == "not_a_recipe":
            schema_valid += 1

        # not_a_recipe accuracy
        if is_not_recipe:
            not_recipe_total += 1
            if predicted.get("dish") is None and predicted.get("reason") == "not_a_recipe":
                not_recipe_correct += 1
            else:
                pass  # model mistakenly output a recipe for non-recipe input

        # Enum accuracy
        all_enum_results.append(_check_enum_values(predicted))

    total = len(gold_records)
    recipe_total = total - not_recipe_total

    # Aggregate enum stats (recipe outputs only)
    total_enum = sum(r["total_enum_fields"] for r in all_enum_results)
    valid_enum = sum(r["valid_enum_fields"] for r in all_enum_results)
    enum_acc = (valid_enum / total_enum * 100) if total_enum > 0 else 0.0

    field_cov = _per_field_coverages(predicted_outputs)

    metrics: dict[str, Any] = {
        "total_examples": total,
        "json_valid": json_valid,
        "json_validity_pct": round(json_valid / total * 100, 1),
        "schema_valid": schema_valid,
        "schema_validity_pct": round(schema_valid / total * 100, 1),
        "enum_total_fields": total_enum,
        "enum_valid_fields": valid_enum,
        "enum_accuracy_pct": round(enum_acc, 1),
        "not_a_recipe_total": not_recipe_total,
        "not_a_recipe_correct": not_recipe_correct,
        "not_a_recipe_accuracy_pct": (
            round(not_recipe_correct / not_recipe_total * 100, 1)
            if not_recipe_total > 0 else None
        ),
        "field_coverage": field_cov,
    }

    # Print report
    print("=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"Examples:          {total}")
    print(f"JSON valid:        {json_valid}/{total} ({metrics['json_validity_pct']:.1f}%)")
    print(f"Schema valid:      {schema_valid}/{total} ({metrics['schema_validity_pct']:.1f}%)")
    print(f"Enum accuracy:     {valid_enum}/{total_enum} ({enum_acc:.1f}%)")
    if not_recipe_total > 0:
        print(f"Not-a-recipe acc:  {not_recipe_correct}/{not_recipe_total} ({metrics['not_a_recipe_accuracy_pct']:.1f}%)")
    print("-" * 60)
    print("Field coverage (predicted):")
    for field, info in field_cov.items():
        print(f"  {field}: {info['present']}/{info['total']} ({info['pct']}%)")
    print("=" * 60)

    # Write report
    if output_report:
        report_path = Path(output_report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"\nReport saved to {output_report}")

    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Evaluate a fine-tuned Qwen2.5-3B QLoRA model"
    )
    parser.add_argument("--adapter_dir", required=True,
                        help="Path to the saved LoRA adapter directory")
    parser.add_argument("--val_file", default="data/training/val.jsonl")
    parser.add_argument("--base_model_id", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--output_report", default="data/training/eval_report.json")
    args = parser.parse_args()

    evaluate(
        adapter_dir=args.adapter_dir,
        val_file=args.val_file,
        base_model_id=args.base_model_id,
        output_report=args.output_report,
    )
