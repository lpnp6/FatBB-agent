"""Convert bootstrap labeled JSONL into Qwen-chat-template Alpaca training data.

Reads ``data/bootstrap/training.jsonl`` and produces an 85/15 train/val split
with each record in this shape::

    {
      "instruction": "...",
      "input": "<markdown>",
      "output": "<compact JSON string>"
    }

The system prompt is loaded from the package resource ``system.txt`` and stored
as a separate field so ``train.py`` can prepend it to the chat template. The
split is stratified by source domain (recipetineats / wellplated).

Usage::

    PYTHONPATH=src python -m labeling_sft.dataset_builder
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from importlib.resources import files
from pathlib import Path
from typing import Any

import random


def _extract_domain(record_id: str) -> str:
    """Extract the source domain from a bootstrap record id.

    ``labeling:corpus:www.recipetineats.com-apple-sauce-comment-page-2-...``
    becomes ``"recipetineats"``.
    """
    parts = record_id.split(":")
    if len(parts) < 3:
        return "unknown"
    slug = parts[2]
    match = re.match(r"(?:www\.)?([a-zA-Z0-9-]+)\.(?:com|net|org)", slug)
    return match.group(1) if match else slug.split("-")[0] if "-" in slug else slug


def _load_system_prompt() -> str:
    return (
        files("labeling_sft")
        .joinpath("system.txt")
        .read_text(encoding="utf-8")
        .strip()
    )


def _validate_output(output: dict[str, Any]) -> list[str]:
    """Check that the output dict has the required top-level shape.

    Returns a list of problems (empty = valid).
    """
    problems: list[str] = []
    if not isinstance(output, dict):
        return ["output is not a dict"]

    if output.get("dish") is None:
        # not_a_recipe shape — check reason field
        if output.get("reason") != "not_a_recipe":
            problems.append("non-recipe output missing reason=not_a_recipe")
        return problems

    # Recipe record
    dish = output.get("dish")
    if not isinstance(dish, dict):
        problems.append("dish is not a dict")
    elif not dish.get("name"):
        problems.append("dish.name is missing or empty")

    ingredients = output.get("ingredients")
    if not isinstance(ingredients, list):
        problems.append("ingredients is not a list")

    return problems


def build_dataset(
    input_path: str = "data/bootstrap/training.jsonl",
    train_path: str = "data/training/train.jsonl",
    val_path: str = "data/training/val.jsonl",
    stats_path: str = "data/training/dataset_stats.json",
    val_split: float = 0.15,
    seed: int = 42,
) -> dict[str, Any]:
    """Convert bootstrap JSONL to Alpaca-format train/val splits.

    Args:
        input_path: Path to ``training.jsonl`` from bootstrap labeling.
        train_path: Where to write training split.
        val_path: Where to write validation split.
        stats_path: Where to write dataset statistics.
        val_split: Fraction of data for validation (default 0.15).
        seed: Random seed for reproducible splits.

    Returns:
        Statistics dict.
    """
    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # --- Load and validate ---------------------------------------------------
    instruction = (
        "Extract structured food knowledge from the following recipe. "
        "Output valid JSON matching the FatBB food knowledge graph schema."
    )
    records: list[dict[str, str]] = []
    skipped = 0
    domain_counts: Counter[str] = Counter()
    recipe_count = 0
    not_recipe_count = 0

    with input_file.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)

            # Resolve output — it is already a dict in training.jsonl
            output_obj = raw.get("output")
            if isinstance(output_obj, str):
                try:
                    output_obj = json.loads(output_obj)
                except json.JSONDecodeError:
                    skipped += 1
                    continue

            problems = _validate_output(output_obj)
            if problems and any("missing" in p or "not a dict" in p for p in problems):
                skipped += 1
                continue

            domain = _extract_domain(raw["id"])
            domain_counts[domain] += 1

            if output_obj.get("dish") is None:
                not_recipe_count += 1
            else:
                recipe_count += 1

            records.append({
                "instruction": instruction,
                "input": raw["input"],
                "output": json.dumps(output_obj, ensure_ascii=False, separators=(",", ":")),
                "_domain": domain,
            })

    if not records:
        raise ValueError("no valid records found in input file")

    # --- Stratified split (manual — avoids sklearn dependency) ---------------
    domains = [r.pop("_domain") for r in records]
    rng = random.Random(seed)

    # Group indices by domain
    domain_indices: dict[str, list[int]] = defaultdict(list)
    for i, d in enumerate(domains):
        domain_indices[d].append(i)

    train_indices: list[int] = []
    val_indices: list[int] = []
    for d, indices in sorted(domain_indices.items()):
        rng.shuffle(indices)
        n_val = max(1, int(len(indices) * val_split + 0.5))
        val_indices.extend(indices[:n_val])
        train_indices.extend(indices[n_val:])

    train = [records[i] for i in sorted(train_indices)]
    val = [records[i] for i in sorted(val_indices)]

    # --- Write outputs -------------------------------------------------------
    for path, subset in [(Path(train_path), train), (Path(val_path), val)]:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for record in subset:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    # --- Statistics -----------------------------------------------------------
    train_domain_counts = Counter(domains[i] for i in train_indices)
    val_domain_counts = Counter(domains[i] for i in val_indices)

    stats: dict[str, Any] = {
        "total_valid_records": len(records),
        "skipped_records": skipped,
        "recipe_count": recipe_count,
        "not_a_recipe_count": not_recipe_count,
        "train_count": len(train),
        "val_count": len(val),
        "val_split": val_split,
        "seed": seed,
        "train_domains": dict(train_domain_counts),
        "val_domains": dict(val_domain_counts),
    }

    stats_path_obj = Path(stats_path)
    stats_path_obj.parent.mkdir(parents=True, exist_ok=True)
    stats_path_obj.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert bootstrap labeled JSONL to Alpaca-format train/val splits"
    )
    parser.add_argument(
        "--input", default="data/bootstrap/training.jsonl",
        help="Path to labeled JSONL (default: data/bootstrap/training.jsonl)",
    )
    parser.add_argument(
        "--output_dir", default="models/qwen2.5-3b-fatbb-v1",
        help="Directory to write train.jsonl, val.jsonl, dataset_stats.json",
    )
    parser.add_argument(
        "--val-split", type=float, default=0.15,
        help="Fraction of data for validation (default: 0.15)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    try:
        stats = build_dataset(
            input_path=args.input,
            train_path=str(out / "train.jsonl"),
            val_path=str(out / "val.jsonl"),
            stats_path=str(out / "dataset_stats.json"),
            val_split=args.val_split,
            seed=args.seed,
        )
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        raise SystemExit(1) from exc

    print(f"Total valid:   {stats['total_valid_records']}")
    print(f"Skipped:       {stats['skipped_records']}")
    print(f"Recipes:       {stats['recipe_count']}")
    print(f"Not-a-recipe:  {stats['not_a_recipe_count']}")
    print(f"Train:         {stats['train_count']}  → {out / 'train.jsonl'}")
    print(f"Val:           {stats['val_count']}  → {out / 'val.jsonl'}")
    print(f"Stats:         {out / 'dataset_stats.json'}")
