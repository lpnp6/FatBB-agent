"""Evaluate a fine-tuned Qwen2.5-3B QLoRA model against the validation set.

Loads the 4-bit base model with the saved LoRA adapter, runs greedy-decoding
inference on every validation example, and reports:

- JSON validity (% of outputs that parse as valid JSON)
- Schema validity (% passing the labeling validator)
- Enum accuracy (% of enum fields matching allowed values)
- Per-field completeness vs the gold labels

Comparison mode (``compare()``) also runs the base model **without** the
LoRA adapter on the same validation set and produces a side-by-side report.

Usage::

    # Evaluate fine-tuned model only
    PYTHONPATH=src python -m labeling_sft.evaluators.qwen \\
        --adapter_dir models/qwen2.5-3b-fatbb-v1 \\
        --val_file data/training/val.jsonl

    # Compare fine-tuned vs base model
    PYTHONPATH=src python -m labeling_sft.evaluators.qwen \\
        --adapter_dir models/qwen2.5-3b-fatbb-v1 \\
        --val_file data/training/val.jsonl \\
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

from labeling.utils.validator import OutputValidator, OutputValidationError
from labeling_sft.contracts import ComparisonReport, EvalReport
from labeling_sft.interfaces.evaluator import BaseEvaluator
from labeling_sft.trainers.qlora import format_example, load_system_prompt

logger = logging.getLogger(__name__)

_validator = OutputValidator()  # shared instance — stateless, thread-safe


# ---------------------------------------------------------------------------
# JSON extraction helper
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


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

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
# Evaluation
# ---------------------------------------------------------------------------

class QwenEvaluator(BaseEvaluator):
    """Evaluate a Qwen2.5-3B model (base or fine-tuned) on labeling data.

    Supports single-model evaluation and base-vs-fine-tuned comparison.
    """

    def __init__(self, base_model_id: str = "Qwen/Qwen2.5-3B-Instruct") -> None:
        self._default_base = base_model_id

    # ── BaseEvaluator implementation ────────────────────────────────────

    def load_model(
        self,
        adapter_dir: str | None,
        base_model_id: str,
        local_files_only: bool = False,
        **kwargs,
    ) -> tuple[Any, Any]:
        """Load 4-bit model, optionally with a LoRA adapter."""
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

        if adapter_dir is not None:
            logger.info("Loading LoRA adapter: %s", adapter_dir)
            model = PeftModel.from_pretrained(model, adapter_dir)

        model.eval()
        return model, tokenizer

    def evaluate(
        self,
        adapter_dir: str | None,
        val_path: str,
        base_model_id: str = "",
        output_report: str | None = None,
        max_samples: int | None = None,
        local_files_only: bool = False,
        **kwargs,
    ) -> EvalReport:
        """Evaluate a single model on the validation set."""
        base = base_model_id or self._default_base
        val_path_obj = Path(val_path)
        if not val_path_obj.exists():
            raise FileNotFoundError(f"Validation file not found: {val_path}")

        with val_path_obj.open("r", encoding="utf-8") as fh:
            gold_records = [json.loads(line) for line in fh if line.strip()]

        total_available = len(gold_records)
        if max_samples is not None and max_samples < len(gold_records):
            gold_records = gold_records[:max_samples]
            logger.info("Limited to first %d samples (%.1f%% of full set)", max_samples,
                        max_samples * 100 / total_available)

        system_prompt = load_system_prompt()
        model, tokenizer = self.load_model(
            adapter_dir, base, local_files_only=local_files_only,
        )

        metrics = self._run_eval_pass(model, tokenizer, gold_records, system_prompt)

        report = EvalReport(
            model_label="Fine-tuned" if adapter_dir else "Base",
            total_examples=metrics["total_examples"],
            json_valid=metrics["json_valid"],
            json_validity_pct=metrics["json_validity_pct"],
            validator_pass=metrics["validator_pass"],
            validator_pass_pct=metrics["validator_pass_pct"],
            enum_valid_fields=metrics["enum_valid_fields"],
            enum_total_fields=metrics["enum_total_fields"],
            enum_accuracy_pct=metrics["enum_accuracy_pct"],
            not_a_recipe_correct=metrics["not_a_recipe_correct"],
            not_a_recipe_total=metrics["not_a_recipe_total"],
            not_a_recipe_accuracy_pct=metrics["not_a_recipe_accuracy_pct"],
            field_coverage=metrics["field_coverage"],
            validator_errors=metrics["validator_errors"],
            raw_metrics=metrics,
        )

        self._print_metrics(metrics, report.model_label)

        if output_report:
            report_path = Path(output_report)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(
                json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(f"\n  Report saved to {output_report}")

        return report

    def compare(
        self,
        adapter_dir: str,
        val_path: str,
        base_model_id: str = "",
        output_report: str | None = None,
        diff_examples: int = 5,
        max_samples: int | None = None,
        local_files_only: bool = False,
        **kwargs,
    ) -> ComparisonReport:
        """Compare base model vs fine-tuned model."""
        import torch

        base = base_model_id or self._default_base
        val_path_obj = Path(val_path)
        if not val_path_obj.exists():
            raise FileNotFoundError(f"Validation file not found: {val_path}")

        with val_path_obj.open("r", encoding="utf-8") as fh:
            gold_records = [json.loads(line) for line in fh if line.strip()]

        total_available = len(gold_records)
        if max_samples is not None and max_samples < len(gold_records):
            gold_records = gold_records[:max_samples]
            logger.info("Limited to first %d samples (%.1f%% of full set)", max_samples,
                        max_samples * 100 / total_available)

        system_prompt = load_system_prompt()

        # ── Phase 1: Base model ─────────────────────────────────────────
        print(f"\n{'=' * 60}")
        print("  Phase 1/2: Evaluating base model (no adapter)")
        print(f"{'=' * 60}")
        base_model, tokenizer = self.load_model(
            None, base, local_files_only=local_files_only,
        )
        base_metrics = self._run_eval_pass(
            base_model, tokenizer, gold_records, system_prompt,
            return_predictions=(diff_examples > 0),
        )

        del base_model
        torch.cuda.empty_cache()

        # ── Phase 2: Fine-tuned model ───────────────────────────────────
        print(f"\n{'=' * 60}")
        print("  Phase 2/2: Evaluating fine-tuned model")
        print(f"{'=' * 60}")
        ft_model, _ = self.load_model(
            adapter_dir, base, local_files_only=local_files_only,
        )
        ft_metrics = self._run_eval_pass(
            ft_model, tokenizer, gold_records, system_prompt,
            return_predictions=(diff_examples > 0),
        )

        # ── Build reports ───────────────────────────────────────────────
        base_report = EvalReport(
            model_label="Base (no adapter)",
            total_examples=base_metrics["total_examples"],
            json_valid=base_metrics["json_valid"],
            json_validity_pct=base_metrics["json_validity_pct"],
            validator_pass=base_metrics["validator_pass"],
            validator_pass_pct=base_metrics["validator_pass_pct"],
            enum_valid_fields=base_metrics["enum_valid_fields"],
            enum_total_fields=base_metrics["enum_total_fields"],
            enum_accuracy_pct=base_metrics["enum_accuracy_pct"],
            not_a_recipe_correct=base_metrics["not_a_recipe_correct"],
            not_a_recipe_total=base_metrics["not_a_recipe_total"],
            not_a_recipe_accuracy_pct=base_metrics["not_a_recipe_accuracy_pct"],
            field_coverage=base_metrics["field_coverage"],
            validator_errors=base_metrics["validator_errors"],
            raw_metrics=base_metrics,
        )
        ft_report = EvalReport(
            model_label="Fine-tuned",
            total_examples=ft_metrics["total_examples"],
            json_valid=ft_metrics["json_valid"],
            json_validity_pct=ft_metrics["json_validity_pct"],
            validator_pass=ft_metrics["validator_pass"],
            validator_pass_pct=ft_metrics["validator_pass_pct"],
            enum_valid_fields=ft_metrics["enum_valid_fields"],
            enum_total_fields=ft_metrics["enum_total_fields"],
            enum_accuracy_pct=ft_metrics["enum_accuracy_pct"],
            not_a_recipe_correct=ft_metrics["not_a_recipe_correct"],
            not_a_recipe_total=ft_metrics["not_a_recipe_total"],
            not_a_recipe_accuracy_pct=ft_metrics["not_a_recipe_accuracy_pct"],
            field_coverage=ft_metrics["field_coverage"],
            validator_errors=ft_metrics["validator_errors"],
            raw_metrics=ft_metrics,
        )

        # ── Print comparison ────────────────────────────────────────────
        print("\n" + "=" * 60)
        print("  Side-by-Side Comparison")
        print("=" * 60)
        self._print_metrics(base_metrics, "Base Model (no adapter)")
        self._print_metrics(ft_metrics, "Fine-tuned Model")
        self._print_comparison(ft_metrics, base_metrics)

        # ── Divergent examples ──────────────────────────────────────────
        divergent: list[dict[str, Any]] = []
        if diff_examples > 0:
            print("\n" + "=" * 60)
            print("  Divergent Examples")
            print("=" * 60)
            base_preds = base_metrics.get("predictions", [])
            ft_preds = ft_metrics.get("predictions", [])
            divergent = self._build_divergent_examples(base_preds, ft_preds, diff_examples)
            self._print_example_diffs(base_preds, ft_preds, diff_examples)

        # ── Write report ────────────────────────────────────────────────
        if output_report:
            comparison = {
                "base_model_id": base,
                "adapter_dir": adapter_dir,
                "base": {k: v for k, v in base_metrics.items() if k != "predictions"},
                "fine_tuned": {k: v for k, v in ft_metrics.items() if k != "predictions"},
            }
            report_path = Path(output_report)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(
                json.dumps(comparison, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(f"\n  Comparison report saved to {output_report}")

        return ComparisonReport(
            base_model_id=base,
            adapter_dir=adapter_dir,
            base=base_report,
            fine_tuned=ft_report,
            divergent_examples=divergent,
        )

    # ── Core eval pass ──────────────────────────────────────────────────

    @staticmethod
    def _run_eval_pass(
        model: Any,
        tokenizer: Any,
        gold_records: list[dict[str, Any]],
        system_prompt: str,
        *,
        return_predictions: bool = False,
    ) -> dict[str, Any]:
        """Run inference on all *gold_records* and return per-metric aggregates."""
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

            generated = QwenEvaluator._generate_one(model, tokenizer, prompt)

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

            # Schema validity
            try:
                _validator.parse(json_text)
                validator_pass += 1
            except OutputValidationError as exc:
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

            # Enum accuracy
            all_enum_results.append(_check_enum_values(predicted))

        total = len(gold_records)

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

    @staticmethod
    def _generate_one(
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

    # ── Display helpers ─────────────────────────────────────────────────

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    @classmethod
    def _print_example_diffs(
        cls,
        base_preds: list[tuple[int, dict[str, Any], str]],
        ft_preds: list[tuple[int, dict[str, Any], str]],
        max_examples: int,
    ) -> None:
        """Print up to *max_examples* side-by-side diffs."""
        base_by_idx: dict[int, dict[str, Any]] = {idx: pred for idx, pred, _raw in base_preds}
        ft_by_idx: dict[int, dict[str, Any]] = {idx: pred for idx, pred, _raw in ft_preds}

        all_indices = sorted(set(base_by_idx) | set(ft_by_idx))
        diffs: list[tuple[int, str, str]] = []

        for idx in all_indices:
            base_out = base_by_idx.get(idx)
            ft_out = ft_by_idx.get(idx)

            if base_out is None and ft_out is not None:
                diffs.append((idx, "(JSON parse failed)", cls._summarize(ft_out)))
            elif base_out is not None and ft_out is None:
                diffs.append((idx, cls._summarize(base_out), "(JSON parse failed)"))
            elif base_out != ft_out:
                diffs.append((idx, cls._summarize(base_out), cls._summarize(ft_out)))

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

    @classmethod
    def _build_divergent_examples(
        cls,
        base_preds: list[tuple[int, dict[str, Any], str]],
        ft_preds: list[tuple[int, dict[str, Any], str]],
        max_examples: int,
    ) -> list[dict[str, Any]]:
        """Build a list of divergent example dicts."""
        base_by_idx: dict[int, dict[str, Any]] = {idx: pred for idx, pred, _raw in base_preds}
        ft_by_idx: dict[int, dict[str, Any]] = {idx: pred for idx, pred, _raw in ft_preds}

        all_indices = sorted(set(base_by_idx) | set(ft_by_idx))
        result: list[dict[str, Any]] = []

        for idx in all_indices:
            base_out = base_by_idx.get(idx)
            ft_out = ft_by_idx.get(idx)
            if base_out != ft_out:
                result.append({
                    "index": idx,
                    "base_summary": cls._summarize(base_out) if base_out else "(JSON parse failed)",
                    "ft_summary": cls._summarize(ft_out) if ft_out else "(JSON parse failed)",
                })

        return result[:max_examples]


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

    evaluator = QwenEvaluator(base_model_id=args.base_model_id)

    if args.compare_base:
        evaluator.compare(
            adapter_dir=args.adapter_dir,
            val_path=args.val_file,
            base_model_id=args.base_model_id,
            output_report="data/training/eval_comparison.json",
            diff_examples=args.diff_examples,
            local_files_only=args.local_files_only,
            max_samples=args.max_samples,
        )
    else:
        evaluator.evaluate(
            adapter_dir=args.adapter_dir,
            val_path=args.val_file,
            base_model_id=args.base_model_id,
            output_report=args.output_report,
            local_files_only=args.local_files_only,
            max_samples=args.max_samples,
        )
