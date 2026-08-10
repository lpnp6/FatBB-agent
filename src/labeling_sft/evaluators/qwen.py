"""Deterministic evaluation for Qwen recipe-extraction models.

The evaluator separates generation from scoring.  Scoring is CPU-only and
compares structured predictions with gold JSON using accuracy and F1 metrics;
it can therefore be tested without loading a model or GPU dependencies.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from collections import Counter
from importlib.resources import files
from pathlib import Path
from typing import Any

from tqdm import tqdm

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
from labeling_sft.trainers.qlora import format_example

logger = logging.getLogger(__name__)
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _adapter_sha256(adapter: Path) -> str:
    files = [adapter / "adapter_config.json"] + [
        path for path in (adapter / "adapter_model.safetensors", adapter / "adapter_model.bin")
        if path.is_file()
    ]
    if len(files) == 1:
        raise FileNotFoundError(f"No adapter weights found in {adapter}")
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode())
        digest.update(_sha256(path).encode())
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    started = time.perf_counter()
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    logger.info("Wrote %s in %.2fs", path, time.perf_counter() - started)


def _load_system_prompt() -> str:
    started = time.perf_counter()
    prompt = files("labeling.prompts").joinpath("system.txt").read_text(encoding="utf-8").strip()
    logger.info("Loaded system prompt (%d chars) in %.2fs", len(prompt), time.perf_counter() - started)
    return prompt


def _configure_logging(path: Path) -> None:
    """Send evaluator logs to both stderr and one run-local file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    for handler in (logging.StreamHandler(sys.stderr), logging.FileHandler(path, encoding="utf-8")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)


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

    def __init__(
        self,
        base_model_id: str = "Qwen/Qwen2.5-3B-Instruct",
        ollama_model: str = "qwen2.5-fatbb:v2",
        ollama_host: str = "http://127.0.0.1:11434",
    ) -> None:
        if not ollama_model:
            raise ValueError("ollama_model must not be empty")
        if not ollama_host.startswith(("http://", "https://")):
            raise ValueError("ollama_host must include http:// or https://")
        self._default_base = base_model_id
        self._ollama_model = ollama_model
        self._ollama_host = ollama_host.rstrip("/")

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

        started = time.perf_counter()
        logger.info("Loading base model %s", training.base_model_id)
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
            logger.info("Loading LoRA adapter from %s", training.adapter.value)
            model = PeftModel.from_pretrained(model, self._local_artifact_path(training.adapter, "adapter"))
        model.eval()
        logger.info("Loaded model include_adapter=%s in %.2fs", include_adapter, time.perf_counter() - started)
        return model, tokenizer

    def evaluate(
        self,
        training: TrainingResult,
        dataset: DatasetSplit,
        report_target: ArtifactLocation | None = None,
        max_samples: int | None = None,
        local_files_only: bool = False,
        work_dir: str | Path | None = None,
        resume: bool = False,
        **kwargs: Any,
    ) -> EvalReport:
        """Evaluate the fine-tuned model, optionally checkpointing in ``work_dir/evaluation``."""
        log_path = self._evaluation_directory(training, work_dir) / "evaluator.log" if work_dir else (
            self._local_artifact_path(report_target, "report_target").parent / "evaluator.log"
            if report_target else Path("evaluator.log")
        )
        _configure_logging(log_path)
        logger.info("Starting evaluation adapter=%s dataset=%s resume=%s", training.adapter.value, dataset.val.value, resume)
        checkpoint = self._prepare_checkpoint(training, dataset, work_dir, max_samples, resume)
        metrics = self._evaluate_metrics(training, dataset, max_samples, local_files_only, checkpoint)
        report = EvalReport("Fine-tuned", metrics["total_examples"], metrics)
        target = report_target or (
            ArtifactLocation.local(str(checkpoint["directory"] / "report.json")) if checkpoint else None
        )
        self._write_report(metrics, target)
        if checkpoint:
            checkpoint["manifest"]["status"] = "completed"
            _atomic_json(checkpoint["manifest_path"], checkpoint["manifest"])
        logger.info("Evaluation completed: %d examples", metrics["total_examples"])
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
        raise NotImplementedError(
            "Base-vs-adapter comparison requires two Ollama model names; "
            f"the Ollama evaluator currently evaluates only {self._ollama_model}."
        )

    def _evaluate_metrics(
        self,
        training: TrainingResult,
        dataset: DatasetSplit,
        max_samples: int | None,
        local_files_only: bool,
        checkpoint: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        records = self._load_records(dataset, max_samples)
        completed = (
            {index: item for index, item in checkpoint["completed"].items() if 0 <= index < len(records)}
            if checkpoint else {}
        )
        metrics, _predictions = self._run_eval_pass(
            None,
            None,
            records,
            _load_system_prompt(),
            completed=completed,
            prediction_path=checkpoint["predictions_path"] if checkpoint else None,
        )
        return metrics

    @staticmethod
    def _evaluation_directory(training: TrainingResult, work_dir: str | Path) -> Path:
        if training.adapter.type is not ArtifactLocationType.LOCAL_PATH:
            raise NotImplementedError("work_dir evaluation requires a local adapter")
        root = Path(work_dir).expanduser().resolve()
        adapter = Path(training.adapter.value).expanduser().resolve()
        return root / "evaluation" if adapter == root else root / "evaluation" / adapter.name

    @staticmethod
    def _load_records(dataset: DatasetSplit, max_samples: int | None) -> list[dict[str, Any]]:
        path = QwenEvaluator._local_data_path(dataset.val, "validation split")
        started = time.perf_counter()
        logger.info("Reading validation dataset from %s", path)
        with path.open(encoding="utf-8") as stream:
            records = [json.loads(line) for line in stream if line.strip()]
        records = records if max_samples is None else records[:max_samples]
        logger.info("Read %d validation records in %.2fs", len(records), time.perf_counter() - started)
        return records

    def _run_eval_pass(
        self,
        model: Any | None,
        tokenizer: Any | None,
        records: list[dict[str, Any]],
        system_prompt: str,
        *,
        completed: dict[int, dict[str, Any]] | None = None,
        prediction_path: Path | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        scored: list[tuple[dict[str, Any], dict[str, Any], bool, bool]] = []
        predictions: list[dict[str, Any]] = []
        completed = completed or {}
        logger.info("Evaluation records: total=%d resumed=%d pending=%d", len(records), len(completed), len(records) - len(completed))
        with tqdm(total=len(records), initial=len(completed), desc="Evaluating", unit="sample") as progress:
            for index, record in enumerate(records):
                gold = json.loads(record["output"])
                prediction = completed.get(index)
                if prediction is None:
                    logger.info("Generating sample %d/%d", index + 1, len(records))
                    prompt = format_example({**record, "output": ""}, system_prompt)
                    prompt = prompt.rsplit("<|im_start|>assistant\n", 1)[0] + "<|im_start|>assistant\n"
                    started = time.perf_counter()
                    raw_output = self._generate_one(None, None, prompt)
                    logger.info("Generated sample %d/%d in %.2fs", index + 1, len(records), time.perf_counter() - started)
                    prediction = QwenEvaluator._prediction_record(index, raw_output)
                    if prediction_path:
                        QwenEvaluator._append_prediction(prediction_path, prediction)
                    progress.update()
                scored.append((gold, prediction["prediction"], prediction["json_valid"], prediction["schema_valid"]))
                predictions.append(prediction)
        return _score_records(scored), predictions

    def _generate_one(self, model: Any, tokenizer: Any, prompt: str, max_new_tokens: int = 4096) -> str:
        """Generate through local Ollama; ``model`` and ``tokenizer`` are unused."""
        payload = json.dumps({
            "model": self._ollama_model,
            "prompt": prompt,
            "raw": True,
            "stream": False,
            "options": {"temperature": 0, "num_ctx": 8192, "num_predict": max_new_tokens},
        }).encode("utf-8")
        request = Request(
            f"{self._ollama_host}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=600) as response:
                result = json.loads(response.read())
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama request failed: HTTP {error.code}: {detail}") from error
        except URLError as error:
            raise RuntimeError(f"Cannot reach Ollama at {self._ollama_host}: {error.reason}") from error
        output = result.get("response")
        if not isinstance(output, str) or not output.strip():
            raise RuntimeError(f"Ollama returned an empty response: {result}")
        eval_duration = result.get("eval_duration", 0) or 0
        eval_count = result.get("eval_count", 0) or 0
        tokens_per_second = eval_count / (eval_duration / 1_000_000_000) if eval_duration else 0.0
        logger.info(
            "Ollama completed model=%s prompt_tokens=%s output_tokens=%s eval_tps=%.1f",
            self._ollama_model, result.get("prompt_eval_count", "?"), eval_count, tokens_per_second,
        )
        return output.strip()

    @staticmethod
    def _prediction_record(index: int, raw_output: str) -> dict[str, Any]:
        json_text = _extract_json(raw_output)
        predicted: dict[str, Any] = {}
        json_valid = schema_valid = False
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
        return {
            "index": index,
            "raw_output": raw_output,
            "prediction": predicted,
            "json_valid": json_valid,
            "schema_valid": schema_valid,
        }

    @staticmethod
    def _append_prediction(path: Path, prediction: dict[str, Any]) -> None:
        started = time.perf_counter()
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(prediction, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        logger.info("Checkpointed sample %d to %s in %.2fs", prediction["index"] + 1, path, time.perf_counter() - started)

    @staticmethod
    def _prepare_checkpoint(
        training: TrainingResult,
        dataset: DatasetSplit,
        work_dir: str | Path | None,
        max_samples: int | None,
        resume: bool,
    ) -> dict[str, Any] | None:
        if work_dir is None:
            if resume:
                raise ValueError("resume requires work_dir")
            return None
        if training.adapter.type is not ArtifactLocationType.LOCAL_PATH:
            raise NotImplementedError("work_dir evaluation requires a local adapter")

        root = Path(work_dir).expanduser().resolve()
        adapter = Path(training.adapter.value).expanduser().resolve()
        data_path = QwenEvaluator._local_data_path(dataset.val, "validation split").resolve()
        directory = QwenEvaluator._evaluation_directory(training, root)
        manifest_path = directory / "manifest.json"
        predictions_path = directory / "predictions.jsonl"
        started = time.perf_counter()
        logger.info("Fingerprinting dataset and adapter for %s", directory)
        manifest = {
            "version": 1,
            "status": "running",
            "dataset_path": str(data_path),
            "dataset_sha256": _sha256(data_path),
            "adapter_path": str(adapter),
            "adapter_sha256": _adapter_sha256(adapter),
            "base_model_id": training.base_model_id,
            "max_samples": max_samples,
        }
        logger.info("Fingerprinting completed in %.2fs", time.perf_counter() - started)
        if manifest_path.exists():
            logger.info("Reading evaluation manifest from %s", manifest_path)
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            expected = {key: value for key, value in manifest.items() if key != "status"}
            actual = {key: existing.get(key) for key in expected}
            if actual != expected:
                raise ValueError("Evaluation checkpoint does not match this dataset, adapter, or configuration")
            if not resume:
                raise FileExistsError(f"Evaluation already exists at {directory}; pass resume=True to continue")
            manifest = existing
        else:
            directory.mkdir(parents=True, exist_ok=True)
            if predictions_path.exists():
                raise ValueError(f"Found predictions without manifest at {directory}")
            _atomic_json(manifest_path, manifest)

        completed: dict[int, dict[str, Any]] = {}
        if predictions_path.exists():
            started = time.perf_counter()
            logger.info("Reading prediction checkpoint from %s", predictions_path)
            with predictions_path.open(encoding="utf-8") as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # a killed process can leave one partial final line
                    if isinstance(item.get("index"), int):
                        completed[item["index"]] = item
            logger.info("Restored %d prediction records in %.2fs", len(completed), time.perf_counter() - started)
        return {
            "directory": directory,
            "manifest_path": manifest_path,
            "manifest": manifest,
            "predictions_path": predictions_path,
            "completed": completed,
        }

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
        started = time.perf_counter()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        logger.info("Wrote evaluation report to %s in %.2fs", path, time.perf_counter() - started)

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
    parser.add_argument("--work_dir", required=True, type=Path, help="Training output directory")
    parser.add_argument("--model_dir", type=Path, help="Adapter/checkpoint to evaluate (default: work_dir)")
    parser.add_argument("--val_file", type=Path, help="Validation JSONL (default: work_dir/Alpaca/val.jsonl)")
    parser.add_argument("--base_model_id", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--ollama_model", default="qwen2.5-fatbb:v2")
    parser.add_argument("--ollama_host", default="http://127.0.0.1:11434")
    parser.add_argument("--compare_base", action="store_true")
    parser.add_argument("--diff_examples", type=int, default=5)
    parser.add_argument("--max_samples", type=int)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    work_dir = args.work_dir.expanduser().resolve()
    model_dir = (args.model_dir or work_dir).expanduser().resolve()
    val_file = (args.val_file or work_dir / "Alpaca" / "val.jsonl").expanduser().resolve()
    training = TrainingResult(
        model=ArtifactLocation.local(str(model_dir)),
        adapter=ArtifactLocation.local(str(model_dir)),
        base_model_id=args.base_model_id,
    )
    dataset = DatasetSplit(DataLocation.local(str(val_file)), DataLocation.local(str(val_file)))
    evaluator = QwenEvaluator(args.base_model_id, args.ollama_model, args.ollama_host)
    if args.compare_base:
        if args.resume:
            parser.error("--resume is not supported with --compare_base")
        target = (
            work_dir / "evaluation" / "comparison.json"
            if model_dir == work_dir else work_dir / "evaluation" / model_dir.name / "comparison.json"
        )
        evaluator.compare(training, dataset, ArtifactLocation.local(str(target)), args.diff_examples,
                          args.max_samples, args.local_files_only)
    else:
        evaluator.evaluate(training, dataset, max_samples=args.max_samples, local_files_only=args.local_files_only,
                           work_dir=work_dir, resume=args.resume)
