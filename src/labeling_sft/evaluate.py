"""Evaluate a fine-tuned Qwen2.5-3B QLoRA model against the validation set.

Loads the 4-bit base model with the saved LoRA adapter, runs greedy-decoding
inference on every validation example, and reports:

- JSON validity (% of outputs that parse as valid JSON)
- Schema validity (% passing the labeling validator)
- Enum accuracy (% of enum fields matching allowed values)
- Per-field completeness vs the gold labels

Comparison mode (``--compare_base``) also runs the base model **without** the
LoRA adapter on the same validation set and produces a side-by-side report.

Usage::

    # Evaluate fine-tuned model only
    PYTHONPATH=src python -m labeling_sft.evaluate \
        --adapter_dir models/qwen2.5-3b-fatbb-v1 \
        --val_file data/training/val.jsonl

    # Compare fine-tuned vs base model
    PYTHONPATH=src python -m labeling_sft.evaluate \
        --adapter_dir models/qwen2.5-3b-fatbb-v1 \
        --val_file data/training/val.jsonl \
        --compare_base
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any

from .train import format_example, load_system_prompt
from labeling.bootstrap.validator import OutputValidator, OutputValidationError

logger = logging.getLogger(__name__)

_validator = OutputValidator()  # shared instance — stateless, thread-safe


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def _load_base_model(
    base_model_id: str = "Qwen/Qwen2.5-3B-Instruct",
    *,
    local_files_only: bool = False,
) -> tuple[Any, Any]:
    """Load the 4-bit base model **without** any LoRA adapter.

    Returns ``(model, tokenizer)``.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    logger.info("Loading base model (no adapter): %s", base_model_id)
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id, trust_remote_code=True, local_files_only=local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    model.eval()
    return model, tokenizer


def load_eval_model(
    adapter_dir: str,
    base_model_id: str = "Qwen/Qwen2.5-3B-Instruct",
    *,
    local_files_only: bool = False,
) -> tuple[Any, Any]:
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
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id, trust_remote_code=True, local_files_only=local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=local_files_only,
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
# Single-model evaluation (shared by base & fine-tuned)
# ---------------------------------------------------------------------------

def _run_eval_pass(
    model: Any,
    tokenizer: Any,
    gold_records: list[dict[str, Any]],
    system_prompt: str,
    *,
    return_predictions: bool = False,
) -> dict[str, Any]:
    """Run inference on all *gold_records* and return per-metric aggregates.

    This is the shared evaluation loop used for both the base model and the
    fine-tuned model so that metrics are computed identically.

    When *return_predictions* is True, the result dict additionally contains
    ``"predictions"`` — a list of ``(index, predicted_dict, raw_generated_text)``
    tuples for every example that produced valid JSON.
    """
    json_valid = 0
    validator_pass = 0
    validator_errors: Counter[str] = Counter()
    not_recipe_correct = 0
    not_recipe_total = 0
    all_enum_results: list[dict[str, int]] = []
    predicted_outputs: list[dict[str, Any]] = []
    per_example: list[tuple[int, dict[str, Any], str]] = []

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
        if return_predictions:
            per_example.append((i, predicted, generated))

        # Schema validity — full OutputValidator check (enums, types, cross-refs, required fields)
        try:
            _validator.parse(json_text)
            validator_pass += 1
        except OutputValidationError as exc:
            # Classify the error for breakdown reporting
            msg = str(exc)
            if "not found" in msg:
                validator_errors["cross_ref_mismatch"] += 1
            elif ": invalid value" in msg:
                validator_errors["invalid_enum"] += 1
            elif "expected" in msg:
                validator_errors["type_mismatch"] += 1
            elif "required" in msg:
                validator_errors["missing_required"] += 1
            elif "reason=not_a_recipe" in msg:
                validator_errors["bad_non_recipe"] += 1
            else:
                validator_errors["other"] += 1

        # not_a_recipe accuracy
        if is_not_recipe:
            not_recipe_total += 1
            if predicted.get("dish") is None and predicted.get("reason") == "not_a_recipe":
                not_recipe_correct += 1

        # Enum accuracy (per-field detail, complementary to validator)
        all_enum_results.append(_check_enum_values(predicted))

    total = len(gold_records)

    # Aggregate enum stats (recipe outputs only)
    total_enum = sum(r["total_enum_fields"] for r in all_enum_results)
    valid_enum = sum(r["valid_enum_fields"] for r in all_enum_results)
    enum_acc = (valid_enum / total_enum * 100) if total_enum > 0 else 0.0

    field_cov = _per_field_coverages(predicted_outputs)

    result: dict[str, Any] = {
        "total_examples": total,
        "json_valid": json_valid,
        "json_validity_pct": round(json_valid / total * 100, 1),
        "validator_pass": validator_pass,
        "validator_pass_pct": round(validator_pass / total * 100, 1),
        "validator_errors": dict(validator_errors),
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
    if return_predictions:
        result["predictions"] = per_example
    return result


def _print_metrics(metrics: dict[str, Any], label: str) -> None:
    """Pretty-print a single-model metrics dict with a header."""
    print(f"\n{'─' * 60}")
    print(f"  {label}")
    print(f"{'─' * 60}")
    print(f"  Examples:           {metrics['total_examples']}")
    print(f"  JSON valid:         {metrics['json_valid']}/{metrics['total_examples']} "
          f"({metrics['json_validity_pct']:.1f}%)")
    print(f"  Validator pass:     {metrics['validator_pass']}/{metrics['total_examples']} "
          f"({metrics['validator_pass_pct']:.1f}%)")
    if metrics.get("validator_errors"):
        print(f"  Validator failures:")
        for err_type, count in sorted(metrics["validator_errors"].items(), key=lambda x: -x[1]):
            pct = round(count / metrics['total_examples'] * 100, 1)
            print(f"    {err_type}: {count} ({pct}%)")
    print(f"  Enum accuracy:      {metrics['enum_valid_fields']}/{metrics['enum_total_fields']} "
          f"({metrics['enum_accuracy_pct']:.1f}%)")
    if metrics["not_a_recipe_total"] > 0:
        na = metrics["not_a_recipe_accuracy_pct"]
        print(f"  Not-a-recipe acc:   {metrics['not_a_recipe_correct']}/"
              f"{metrics['not_a_recipe_total']} ({na:.1f}%)")
    print(f"  Field coverage:")
    for field, info in metrics["field_coverage"].items():
        print(f"    {field}: {info['present']}/{info['total']} ({info['pct']}%)")


def _print_example_diffs(
    base_preds: list[tuple[int, dict[str, Any], str]],
    ft_preds: list[tuple[int, dict[str, Any], str]],
    max_examples: int,
) -> None:
    """Print up to *max_examples* side-by-side diffs where the two models differ.

    "Differ" means: both models produced valid JSON, but the parsed dicts are
    not equal, OR one model succeeded and the other failed JSON parsing.
    """
    base_by_idx: dict[int, dict[str, Any]] = {idx: pred for idx, pred, _raw in base_preds}
    ft_by_idx: dict[int, dict[str, Any]] = {idx: pred for idx, pred, _raw in ft_preds}

    all_indices = sorted(set(base_by_idx) | set(ft_by_idx))
    diffs: list[tuple[int, str, str]] = []

    for idx in all_indices:
        base_out = base_by_idx.get(idx)
        ft_out = ft_by_idx.get(idx)

        if base_out is None and ft_out is not None:
            diffs.append((idx, "(JSON parse failed)", _summarize(ft_out)))
        elif base_out is not None and ft_out is None:
            diffs.append((idx, _summarize(base_out), "(JSON parse failed)"))
        elif base_out != ft_out:
            diffs.append((idx, _summarize(base_out), _summarize(ft_out)))

    if not diffs:
        print("  (No differences found — both models produced identical outputs on all examples)")
        return

    shown = diffs[:max_examples]
    print(f"\n  Showing {len(shown)} of {len(diffs)} divergent examples:")
    for idx, base_summary, ft_summary in shown:
        print(f"  ┌─ Example #{idx + 1}")
        print(f"  │  Base:     {base_summary}")
        print(f"  │  Fine-tuned: {ft_summary}")
        print(f"  └─")


def _summarize(pred: dict[str, Any]) -> str:
    """One-line summary of a predicted labeling dict."""
    dish = pred.get("dish")
    if dish is None:
        reason = pred.get("reason", "?")
        return f"not_a_recipe (reason={reason})"
    name = dish.get("name", "?")
    dish_type = dish.get("dish_type", "?")
    n_ingredients = len(pred.get("ingredients") or [])
    n_steps = len(dish.get("cooking_steps") or [])
    return f"dish={name!r}, type={dish_type}, {n_ingredients} ingredients, {n_steps} steps"


def _print_comparison(finetuned: dict[str, Any], base: dict[str, Any]) -> None:
    """Print a side-by-side comparison table with deltas."""
    def _delta(ft_val: float | int, base_val: float | int) -> str:
        diff = ft_val - base_val
        sign = "+" if diff >= 0 else ""
        return f"{sign}{diff:.1f}" if isinstance(diff, float) else f"{sign}{diff}"

    rows: list[tuple[str, str, str, str]] = []

    rows.append(("JSON validity %",
                 f"{finetuned['json_validity_pct']:.1f}%",
                 f"{base['json_validity_pct']:.1f}%",
                 _delta(finetuned['json_validity_pct'], base['json_validity_pct'])))
    rows.append(("Validator pass %",
                 f"{finetuned['validator_pass_pct']:.1f}%",
                 f"{base['validator_pass_pct']:.1f}%",
                 _delta(finetuned['validator_pass_pct'], base['validator_pass_pct'])))
    rows.append(("Enum accuracy %",
                 f"{finetuned['enum_accuracy_pct']:.1f}%",
                 f"{base['enum_accuracy_pct']:.1f}%",
                 _delta(finetuned['enum_accuracy_pct'], base['enum_accuracy_pct'])))
    if finetuned["not_a_recipe_total"] > 0:
        rows.append(("Not-a-recipe acc %",
                     f"{finetuned['not_a_recipe_accuracy_pct']:.1f}%",
                     f"{base['not_a_recipe_accuracy_pct']:.1f}%",
                     _delta(finetuned['not_a_recipe_accuracy_pct'],
                            base['not_a_recipe_accuracy_pct'])))

    # Per-field coverage comparison
    all_fields = set(finetuned["field_coverage"]) | set(base["field_coverage"])
    for field in sorted(all_fields):
        ft_cov = finetuned["field_coverage"].get(field, {})
        base_cov = base["field_coverage"].get(field, {})
        ft_pct = ft_cov.get("pct", 0)
        base_pct = base_cov.get("pct", 0)
        rows.append((f"  field: {field}",
                     f"{ft_pct}%",
                     f"{base_pct}%",
                     _delta(ft_pct, base_pct)))

    # Render table
    col_widths = [32, 16, 16, 10]
    header = ["Metric", "Fine-tuned", "Base", "Δ"]
    sep = "─" * (sum(col_widths) + 9)

    print(f"\n{sep}")
    print(f"  {header[0]:<{col_widths[0]}} │ {header[1]:>{col_widths[1]}} │ "
          f"{header[2]:>{col_widths[2]}} │ {header[3]:>{col_widths[3]}}")
    print(f"  {'─' * col_widths[0]}─┼─{'─' * col_widths[1]}─┼─"
          f"{'─' * col_widths[2]}─┼─{'─' * col_widths[3]}")
    for label, ft_val, base_val, delta in rows:
        delta_str = f"\033[92m{delta}\033[0m" if delta.startswith("+") else (
            f"\033[91m{delta}\033[0m" if delta.startswith("-") else delta)
        print(f"  {label:<{col_widths[0]}} │ {ft_val:>{col_widths[1]}} │ "
              f"{base_val:>{col_widths[2]}} │ {delta_str:>{col_widths[3]}}")
    print(f"{sep}\n")


# ---------------------------------------------------------------------------
# Main evaluation entry points
# ---------------------------------------------------------------------------

def evaluate(
    adapter_dir: str,
    val_file: str = "data/training/val.jsonl",
    base_model_id: str = "Qwen/Qwen2.5-3B-Instruct",
    output_report: str | None = "data/training/eval_report.json",
    *,
    local_files_only: bool = False,
    max_samples: int | None = None,
) -> dict[str, Any]:
    """Run the evaluation on the fine-tuned model only and return a metrics dict.

    *max_samples* limits evaluation to the first N records (None = all).
    """
    val_path = Path(val_file)
    if not val_path.exists():
        raise FileNotFoundError(f"Validation file not found: {val_file}")

    with val_path.open("r", encoding="utf-8") as fh:
        gold_records = [json.loads(line) for line in fh if line.strip()]

    if max_samples is not None and max_samples < len(gold_records):
        gold_records = gold_records[:max_samples]
        logger.info("Limited to first %d samples (%.1f%% of full set)", max_samples,
                    max_samples * 100 / len(gold_records))

    system_prompt = load_system_prompt()
    model, tokenizer = load_eval_model(adapter_dir, base_model_id, local_files_only=local_files_only)

    metrics = _run_eval_pass(model, tokenizer, gold_records, system_prompt)

    # Print report
    print("\n" + "=" * 60)
    print("  Evaluation Results — Fine-tuned Model")
    print("=" * 60)
    _print_metrics(metrics, "Fine-tuned")

    # Write report
    if output_report:
        report_path = Path(output_report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"\n  Report saved to {output_report}")

    return metrics


def evaluate_with_comparison(
    adapter_dir: str,
    val_file: str = "data/training/val.jsonl",
    base_model_id: str = "Qwen/Qwen2.5-3B-Instruct",
    output_report: str | None = "data/training/eval_comparison.json",
    diff_examples: int = 5,
    *,
    local_files_only: bool = False,
    max_samples: int | None = None,
) -> dict[str, Any]:
    """Evaluate both the fine-tuned and base model, then compare.

    Runs the base model **first** (without adapter), then the fine-tuned model,
    and produces a side-by-side comparison report.

    When *diff_examples* > 0, also prints up to that many examples where the
    two models produced different outputs.

    *max_samples* limits evaluation to the first N records (None = all).
    """
    import torch

    val_path = Path(val_file)
    if not val_path.exists():
        raise FileNotFoundError(f"Validation file not found: {val_file}")

    with val_path.open("r", encoding="utf-8") as fh:
        gold_records = [json.loads(line) for line in fh if line.strip()]

    if max_samples is not None and max_samples < len(gold_records):
        gold_records = gold_records[:max_samples]
        logger.info("Limited to first %d samples (%.1f%% of full set)", max_samples,
                    max_samples * 100 / len(gold_records))

    system_prompt = load_system_prompt()

    # ── Base model pass ───────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  Phase 1/2: Evaluating base model (no adapter)")
    print(f"{'=' * 60}")
    base_model, tokenizer = _load_base_model(base_model_id, local_files_only=local_files_only)
    base_metrics = _run_eval_pass(
        base_model, tokenizer, gold_records, system_prompt,
        return_predictions=(diff_examples > 0),
    )

    # Free base model VRAM before loading adapter
    del base_model
    torch.cuda.empty_cache()

    # ── Fine-tuned model pass ─────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  Phase 2/2: Evaluating fine-tuned model")
    print(f"{'=' * 60}")
    ft_model, _ = load_eval_model(adapter_dir, base_model_id, local_files_only=local_files_only)
    ft_metrics = _run_eval_pass(
        ft_model, tokenizer, gold_records, system_prompt,
        return_predictions=(diff_examples > 0),
    )

    # ── Print comparison ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Side-by-Side Comparison")
    print("=" * 60)
    _print_metrics(base_metrics, "Base Model (no adapter)")
    _print_metrics(ft_metrics, "Fine-tuned Model")
    _print_comparison(ft_metrics, base_metrics)

    # ── Qualitative diff examples ─────────────────────────────────────────
    if diff_examples > 0:
        print("\n" + "=" * 60)
        print("  Divergent Examples")
        print("=" * 60)
        _print_example_diffs(
            base_metrics.get("predictions", []),
            ft_metrics.get("predictions", []),
            diff_examples,
        )

    # ── Write report ──────────────────────────────────────────────────────
    comparison = {
        "base_model_id": base_model_id,
        "adapter_dir": adapter_dir,
        "base": {k: v for k, v in base_metrics.items() if k != "predictions"},
        "fine_tuned": {k: v for k, v in ft_metrics.items() if k != "predictions"},
    }
    if output_report:
        report_path = Path(output_report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(comparison, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"\n  Comparison report saved to {output_report}")

    return comparison


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
    parser.add_argument(
        "--compare_base", action="store_true",
        help="Also evaluate the base model (no adapter) and produce a side-by-side comparison",
    )
    parser.add_argument(
        "--diff_examples", type=int, default=5,
        help="Number of divergent examples to print when --compare_base is used (default: 5, 0 disables)",
    )
    parser.add_argument(
        "--local_files_only", action="store_true",
        help="Only use locally cached model files; do not attempt to connect to Hugging Face",
    )
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Limit evaluation to the first N validation samples (None = all)",
    )
    args = parser.parse_args()

    if args.compare_base:
        evaluate_with_comparison(
            adapter_dir=args.adapter_dir,
            val_file=args.val_file,
            base_model_id=args.base_model_id,
            output_report="data/training/eval_comparison.json",
            diff_examples=args.diff_examples,
            local_files_only=args.local_files_only,
            max_samples=args.max_samples,
        )
    else:
        evaluate(
            adapter_dir=args.adapter_dir,
            val_file=args.val_file,
            base_model_id=args.base_model_id,
            output_report=args.output_report,
            local_files_only=args.local_files_only,
            max_samples=args.max_samples,
        )
