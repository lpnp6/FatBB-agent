"""Deterministic evaluation for Qwen recipe-extraction models.

The evaluator separates generation from scoring.  Scoring is CPU-only and
compares structured predictions with gold JSON using accuracy and F1 metrics;
it can therefore be tested without loading a model or GPU dependencies.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from labeling.utils.validator import OutputValidationError, OutputValidator
from labeling_sft.contracts import (
    ArtifactLocation,
    ArtifactLocationType,
    ComparisonReport,
    DataLocation,
    DataLocationType,
    DatasetSplit,
    EvalReport,
    TrainingResult,
)
from labeling_sft.interfaces.evaluator import BaseEvaluator
from labeling_sft.trainers.qlora import format_example, load_system_prompt

_VALIDATOR = OutputValidator(mode="finetune")
_SCALAR_FIELDS = (
    ("name", ("dish", "name")),
    ("dish_type", ("dish", "dish_type")),
    ("difficulty", ("dish", "difficulty")),
    ("cuisine", ("dish", "cuisine", "name")),
    ("prep_time", ("dish", "prep_time_min")),
    ("cooking_time", ("dish", "cooking_time_min")),
    ("total_time", ("dish", "total_time_min")),
    ("servings", ("dish", "servings")),
    ("calories", ("dish", "calories_per_serving")),
)


def _extract_json(text: str) -> str | None:
    """Return one JSON object from a model response, if present."""
    text = re.sub(r"<\|im_end\|>.*$", "", text, flags=re.DOTALL).strip()
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        return fenced.group(1)
    object_match = re.search(r"\{.*\}", text, re.DOTALL)
    return object_match.group(0) if object_match else None


def _normal(value: Any) -> Any:
    """Normalize harmless text differences while retaining structured meaning."""
    if isinstance(value, str):
        return " ".join(value.casefold().split())
    if isinstance(value, list):
        return [_normal(item) for item in value]
    if isinstance(value, dict):
        return {key: _normal(item) for key, item in value.items()}
    return value


def _path(value: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = value
    for part in path:
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _f1(counts: Counter[str], name: str) -> dict[str, float | int]:
    tp, fp, fn = (counts[f"{name}_{key}"] for key in ("tp", "fp", "fn"))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "precision_pct": round(precision * 100, 1),
        "recall_pct": round(recall * 100, 1),
        "f1_pct": round(2 * precision * recall / (precision + recall) * 100, 1)
        if precision + recall else 0.0,
        "support": tp + fn,
    }


def _accuracy(counts: Counter[str], name: str) -> dict[str, float | int]:
    correct, total = counts[f"{name}_correct"], counts[f"{name}_total"]
    return {
        "accuracy_pct": round(correct / total * 100, 1) if total else 0.0,
        "correct": correct,
        "support": total,
    }


def _add_set_score(counts: Counter[str], name: str, predicted: set[Any], gold: set[Any]) -> None:
    counts[f"{name}_tp"] += len(predicted & gold)
    counts[f"{name}_fp"] += len(predicted - gold)
    counts[f"{name}_fn"] += len(gold - predicted)


def _is_non_recipe(value: dict[str, Any]) -> bool:
    return value.get("dish") is None and value.get("reason") == "not_a_recipe"


def _ingredient_map(value: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for ingredient in value.get("ingredients") or []:
        if isinstance(ingredient, dict) and ingredient.get("name"):
            result[str(_normal(ingredient["name"]))] = ingredient
    return result


def _step_pairs(value: dict[str, Any], *, refs: bool = False) -> set[tuple[Any, ...]]:
    dish = value.get("dish")
    if not isinstance(dish, dict):
        return set()
    pairs: set[tuple[Any, ...]] = set()
    for step in dish.get("cooking_steps") or []:
        if not isinstance(step, dict):
            continue
        order = step.get("order")
        if refs:
            pairs.update((order, _normal(ref)) for ref in step.get("ingredient_refs") or [])
        elif step.get("method") is not None:
            pairs.add((order, _normal(step["method"])))
    return pairs


def _relation_triples(value: dict[str, Any]) -> set[tuple[Any, ...]]:
    triples: set[tuple[Any, ...]] = set()
    for relation in (value.get("ingredient_relations") or []) + (value.get("dish_relations") or []):
        if not isinstance(relation, dict):
            continue
        source = relation.get("from_ingredient", relation.get("from_dish"))
        target = relation.get("to_ingredient", relation.get("to_dish"))
        kind = relation.get("relation")
        if source and target and kind:
            triples.add((_normal(source), _normal(kind), _normal(target)))
    return triples


def _score_record(gold: dict[str, Any], predicted: dict[str, Any], counts: Counter[str]) -> None:
    """Accumulate deterministic structured-extraction metrics for one record."""
    gold_non_recipe = _is_non_recipe(gold)
    predicted_non_recipe = _is_non_recipe(predicted)
    counts["non_recipe_tp"] += int(gold_non_recipe and predicted_non_recipe)
    counts["non_recipe_fp"] += int(not gold_non_recipe and predicted_non_recipe)
    counts["non_recipe_fn"] += int(gold_non_recipe and not predicted_non_recipe)
    counts["exact_match_correct"] += int(_normal(gold) == _normal(predicted))
    counts["exact_match_total"] += 1

    if gold_non_recipe:
        return

    for _name, field_path in _SCALAR_FIELDS:
        counts["scalar_correct"] += int(_normal(_path(gold, field_path)) == _normal(_path(predicted, field_path)))
        counts["scalar_total"] += 1

    gold_dish = gold.get("dish") if isinstance(gold.get("dish"), dict) else {}
    predicted_dish = predicted.get("dish") if isinstance(predicted.get("dish"), dict) else {}
    for field in ("dietary", "taste_profile"):
        gold_tags = {_normal(tag) for tag in gold_dish.get(field) or []}
        predicted_tags = {_normal(tag) for tag in predicted_dish.get(field) or []}
        _add_set_score(counts, "tag", predicted_tags, gold_tags)

    gold_ingredients, predicted_ingredients = _ingredient_map(gold), _ingredient_map(predicted)
    _add_set_score(counts, "ingredient", set(predicted_ingredients), set(gold_ingredients))
    for name in set(gold_ingredients) & set(predicted_ingredients):
        for field in ("amount", "category", "preparation", "is_essential"):
            counts["ingredient_attribute_correct"] += int(
                _normal(gold_ingredients[name].get(field)) == _normal(predicted_ingredients[name].get(field))
            )
            counts["ingredient_attribute_total"] += 1

    _add_set_score(counts, "step_method", _step_pairs(predicted), _step_pairs(gold))
    _add_set_score(counts, "step_ingredient_ref", _step_pairs(predicted, refs=True), _step_pairs(gold, refs=True))
    _add_set_score(counts, "relation", _relation_triples(predicted), _relation_triples(gold))


def _score_records(records: list[tuple[dict[str, Any], dict[str, Any], bool, bool]]) -> dict[str, Any]:
    """Return evaluation metrics for ``(gold, prediction, json_valid, schema_valid)`` records."""
    counts: Counter[str] = Counter()
    for gold, predicted, json_valid, schema_valid in records:
        counts["json_valid"] += int(json_valid)
        counts["schema_valid"] += int(schema_valid)
        _score_record(gold, predicted, counts)

    total = len(records)
    return {
        "total_examples": total,
        "json_validity_pct": round(counts["json_valid"] / total * 100, 1) if total else 0.0,
        "schema_validity_pct": round(counts["schema_valid"] / total * 100, 1) if total else 0.0,
        "exact_match": _accuracy(counts, "exact_match"),
        "non_recipe": _f1(counts, "non_recipe"),
        "scalar_fields": _accuracy(counts, "scalar"),
        "tags": _f1(counts, "tag"),
        "ingredients": _f1(counts, "ingredient"),
        "ingredient_attributes": _accuracy(counts, "ingredient_attribute"),
        "step_methods": _f1(counts, "step_method"),
        "step_ingredient_refs": _f1(counts, "step_ingredient_ref"),
        "relations": _f1(counts, "relation"),
    }


class QwenEvaluator(BaseEvaluator):
    """Evaluate a Qwen QLoRA adapter with deterministic recipe metrics."""

    def __init__(self, base_model_id: str = "Qwen/Qwen2.5-3B-Instruct") -> None:
        self._default_base = base_model_id

    def load_model(
        self,
        training: TrainingResult,
        include_adapter: bool = True,
        local_files_only: bool = False,
        **kwargs: Any,
    ) -> tuple[Any, Any]:
        """Load the 4-bit base model and optional LoRA adapter."""
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            training.base_model_id, trust_remote_code=True, local_files_only=local_files_only,
        )
        tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            training.base_model_id,
            quantization_config=quantization,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
        if include_adapter:
            model = PeftModel.from_pretrained(model, self._local_artifact_path(training.adapter, "adapter"))
        model.eval()
        return model, tokenizer

    def evaluate(
        self,
        training: TrainingResult,
        dataset: DatasetSplit,
        report_target: ArtifactLocation | None = None,
        max_samples: int | None = None,
        local_files_only: bool = False,
        **kwargs: Any,
    ) -> EvalReport:
        """Evaluate the fine-tuned model on the validation split."""
        metrics = self._evaluate_metrics(training, dataset, max_samples, local_files_only)
        report = EvalReport("Fine-tuned", metrics["total_examples"], metrics)
        self._write_report(metrics, report_target)
        self._print_metrics(metrics, report.model_label)
        return report

    def compare(
        self,
        training: TrainingResult,
        dataset: DatasetSplit,
        report_target: ArtifactLocation | None = None,
        diff_examples: int = 5,
        max_samples: int | None = None,
        local_files_only: bool = False,
        **kwargs: Any,
    ) -> ComparisonReport:
        """Compare base and fine-tuned models using the same deterministic metrics."""
        import torch

        records = self._load_records(dataset, max_samples)
        system_prompt = load_system_prompt()
        base_model, tokenizer = self.load_model(training, include_adapter=False, local_files_only=local_files_only)
        base_metrics, base_predictions = self._run_eval_pass(base_model, tokenizer, records, system_prompt)
        del base_model
        torch.cuda.empty_cache()

        fine_tuned_model, tokenizer = self.load_model(training, local_files_only=local_files_only)
        fine_tuned_metrics, fine_tuned_predictions = self._run_eval_pass(
            fine_tuned_model, tokenizer, records, system_prompt,
        )
        del fine_tuned_model
        torch.cuda.empty_cache()

        self._write_report(
            {"base_model_id": training.base_model_id, "base": base_metrics, "fine_tuned": fine_tuned_metrics},
            report_target,
        )
        self._print_metrics(base_metrics, "Base")
        self._print_metrics(fine_tuned_metrics, "Fine-tuned")
        divergent = self._divergent_examples(base_predictions, fine_tuned_predictions, diff_examples)
        return ComparisonReport(
            base_model_id=training.base_model_id,
            adapter=training.adapter,
            base=EvalReport("Base", base_metrics["total_examples"], base_metrics),
            fine_tuned=EvalReport("Fine-tuned", fine_tuned_metrics["total_examples"], fine_tuned_metrics),
            divergent_examples=divergent,
        )

    def _evaluate_metrics(
        self,
        training: TrainingResult,
        dataset: DatasetSplit,
        max_samples: int | None,
        local_files_only: bool,
    ) -> dict[str, Any]:
        records = self._load_records(dataset, max_samples)
        model, tokenizer = self.load_model(training, local_files_only=local_files_only)
        metrics, _predictions = self._run_eval_pass(model, tokenizer, records, load_system_prompt())
        return metrics

    @staticmethod
    def _load_records(dataset: DatasetSplit, max_samples: int | None) -> list[dict[str, Any]]:
        path = QwenEvaluator._local_data_path(dataset.val, "validation split")
        with path.open(encoding="utf-8") as stream:
            records = [json.loads(line) for line in stream if line.strip()]
        return records if max_samples is None else records[:max_samples]

    @staticmethod
    def _run_eval_pass(
        model: Any,
        tokenizer: Any,
        records: list[dict[str, Any]],
        system_prompt: str,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        scored: list[tuple[dict[str, Any], dict[str, Any], bool, bool]] = []
        predictions: list[dict[str, Any]] = []
        for index, record in enumerate(records):
            gold = json.loads(record["output"])
            prompt = format_example({**record, "output": ""}, system_prompt)
            prompt = prompt.rsplit("<|im_start|>assistant\n", 1)[0] + "<|im_start|>assistant\n"
            raw_output = QwenEvaluator._generate_one(model, tokenizer, prompt)
            json_text = _extract_json(raw_output)
            predicted: dict[str, Any] = {}
            json_valid = False
            schema_valid = False
            if json_text:
                try:
                    parsed = json.loads(json_text)
                    if isinstance(parsed, dict):
                        predicted, json_valid = parsed, True
                        try:
                            _VALIDATOR.parse(json_text)
                            schema_valid = True
                        except OutputValidationError:
                            pass
                except json.JSONDecodeError:
                    pass
            scored.append((gold, predicted, json_valid, schema_valid))
            predictions.append({"index": index, "prediction": predicted, "raw_output": raw_output})
        return _score_records(scored), predictions

    @staticmethod
    def _generate_one(model: Any, tokenizer: Any, prompt: str, max_new_tokens: int = 4096) -> str:
        import torch

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=8192 - max_new_tokens)
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    @staticmethod
    def _local_data_path(location: DataLocation, name: str) -> Path:
        if location.type is not DataLocationType.LOCAL_PATH:
            raise NotImplementedError(f"QwenEvaluator only supports local JSONL; {name} uses {location.type.value}")
        return Path(location.value)

    @staticmethod
    def _local_artifact_path(location: ArtifactLocation, name: str) -> Path:
        if location.type is not ArtifactLocationType.LOCAL_PATH:
            raise NotImplementedError(f"QwenEvaluator only supports local artifacts; {name} uses {location.type.value}")
        return Path(location.value)

    @staticmethod
    def _write_report(metrics: dict[str, Any], target: ArtifactLocation | None) -> None:
        if not target:
            return
        path = QwenEvaluator._local_artifact_path(target, "report_target")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    @staticmethod
    def _print_metrics(metrics: dict[str, Any], label: str) -> None:
        print(f"\n{label}: {metrics['total_examples']} examples")
        print(f"  JSON/schema validity: {metrics['json_validity_pct']:.1f}% / {metrics['schema_validity_pct']:.1f}%")
        for name in ("exact_match", "non_recipe", "scalar_fields", "tags", "ingredients", "ingredient_attributes", "step_methods", "step_ingredient_refs", "relations"):
            value = metrics[name]
            score = value.get("f1_pct", value.get("accuracy_pct", 0.0))
            print(f"  {name}: {score:.1f}% (n={value['support']})")

    @staticmethod
    def _divergent_examples(
        base: list[dict[str, Any]], fine_tuned: list[dict[str, Any]], limit: int,
    ) -> list[dict[str, Any]]:
        return [
            {
                "index": before["index"],
                "base_prediction": before["prediction"],
                "fine_tuned_prediction": after["prediction"],
            }
            for before, after in zip(base, fine_tuned)
            if before["prediction"] != after["prediction"]
        ][:limit]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter_dir", required=True)
    parser.add_argument("--val_file", default="data/training/val.jsonl")
    parser.add_argument("--base_model_id", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--output_report", default="data/training/eval_report.json")
    parser.add_argument("--compare_base", action="store_true")
    parser.add_argument("--diff_examples", type=int, default=5)
    parser.add_argument("--max_samples", type=int)
    parser.add_argument("--local_files_only", action="store_true")
    args = parser.parse_args()
    training = TrainingResult(
        model=ArtifactLocation.local(args.adapter_dir),
        adapter=ArtifactLocation.local(args.adapter_dir),
        base_model_id=args.base_model_id,
    )
    dataset = DatasetSplit(DataLocation.local(args.val_file), DataLocation.local(args.val_file))
    evaluator = QwenEvaluator(args.base_model_id)
    if args.compare_base:
        evaluator.compare(training, dataset, ArtifactLocation.local(args.output_report), args.diff_examples,
                          args.max_samples, args.local_files_only)
    else:
        evaluator.evaluate(training, dataset, ArtifactLocation.local(args.output_report), max_samples=args.max_samples,
                           local_files_only=args.local_files_only)
