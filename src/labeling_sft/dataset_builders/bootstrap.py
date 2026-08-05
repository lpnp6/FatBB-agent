"""Bootstrap JSONL → Alpaca-format train/val dataset builder.

Reads ``data/bootstrap/training.jsonl`` and produces an 85/15 train/val split
with each record in Alpaca chat-template shape::

    {
      "instruction": "...",
      "input": "<markdown>",
      "output": "<compact JSON string>"
    }

The split is stratified by source domain (recipetineats / wellplated).

Usage::

    PYTHONPATH=src python -m labeling_sft.dataset_builders.bootstrap
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from importlib.resources import files
from pathlib import Path
from typing import Any

import random

from labeling_sft.contracts import (
    DataLocation,
    DataLocationType,
    DatasetBuildRequest,
    DatasetSplit,
    DatasetStats,
)
from labeling_sft.interfaces.dataset_builder import BaseDatasetBuilder


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


class BootstrapDatasetBuilder(BaseDatasetBuilder):
    """Build Alpaca-format train/val splits from bootstrap-labeled JSONL.

    Uses stratified splitting by source domain to ensure each domain
    contributes proportionally to both train and val sets.
    """

    def __init__(self, instruction: str | None = None) -> None:
        self._instruction = instruction or (
            "Extract structured food knowledge from the following recipe. "
            "Output valid JSON matching the FatBB food knowledge graph schema."
        )

    # ── BaseDatasetBuilder implementation ────────────────────────────────

    def build(
        self,
        request: DatasetBuildRequest,
    ) -> DatasetSplit:
        """Convert bootstrap JSONL to Alpaca-format train/val splits."""
        input_file = self._local_path(request.source, "source")
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {request.source.value}")
        train_path = self._local_path(request.train_target, "train_target")
        val_path = self._local_path(request.val_target, "val_target")
        stats_path = (
            self._local_path(request.stats_target, "stats_target")
            if request.stats_target is not None
            else None
        )

        # --- Load and validate -------------------------------------------------
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
                    "instruction": self._instruction,
                    "input": raw["input"],
                    "output": json.dumps(output_obj, ensure_ascii=False, separators=(",", ":")),
                    "_domain": domain,
                })

        if not records:
            raise ValueError("no valid records found in input file")

        # --- Stratified split (manual — avoids sklearn dependency) -------------
        domains = [r.pop("_domain") for r in records]
        rng = random.Random(request.seed)

        # Group indices by domain
        domain_indices: dict[str, list[int]] = defaultdict(list)
        for i, d in enumerate(domains):
            domain_indices[d].append(i)

        train_indices: list[int] = []
        val_indices: list[int] = []
        for d, indices in sorted(domain_indices.items()):
            rng.shuffle(indices)
            n_val = max(1, int(len(indices) * request.val_split + 0.5))
            val_indices.extend(indices[:n_val])
            train_indices.extend(indices[n_val:])

        train = [records[i] for i in sorted(train_indices)]
        val = [records[i] for i in sorted(val_indices)]

        # --- Write outputs -----------------------------------------------------
        for path, subset in [(train_path, train), (val_path, val)]:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as fh:
                for record in subset:
                    fh.write(json.dumps(record, ensure_ascii=False) + "\n")

        # --- Statistics ---------------------------------------------------------
        train_domain_counts = Counter(domains[i] for i in train_indices)
        val_domain_counts = Counter(domains[i] for i in val_indices)

        domain_dist: dict[str, dict[str, int]] = {}
        all_domains = set(train_domain_counts) | set(val_domain_counts)
        for d in sorted(all_domains):
            domain_dist[d] = {
                "train": train_domain_counts.get(d, 0),
                "val": val_domain_counts.get(d, 0),
            }

        stats = DatasetStats(
            total_valid_records=len(records),
            skipped_records=skipped,
            recipe_count=recipe_count,
            not_a_recipe_count=not_recipe_count,
            train_count=len(train),
            val_count=len(val),
            val_split=request.val_split,
            seed=request.seed,
            domain_distribution=domain_dist,
        )

        if stats_path is not None:
            stats_path.parent.mkdir(parents=True, exist_ok=True)
            stats_path.write_text(
                json.dumps(
                    {
                        "total_valid_records": stats.total_valid_records,
                        "skipped_records": stats.skipped_records,
                        "recipe_count": stats.recipe_count,
                        "not_a_recipe_count": stats.not_a_recipe_count,
                        "train_count": stats.train_count,
                        "val_count": stats.val_count,
                        "val_split": stats.val_split,
                        "seed": stats.seed,
                        "train_domains": dict(train_domain_counts),
                        "val_domains": dict(val_domain_counts),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

        return DatasetSplit(
            train=request.train_target,
            val=request.val_target,
            stats=stats,
        )

    @staticmethod
    def _local_path(location: DataLocation, name: str) -> Path:
        if location.type is not DataLocationType.LOCAL_PATH:
            raise NotImplementedError(
                f"BootstrapDatasetBuilder only supports local paths; "
                f"{name} uses {location.type.value}"
            )
        return Path(location.value)


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

    builder = BootstrapDatasetBuilder()
    try:
        split = builder.build(DatasetBuildRequest(
            source=DataLocation.local(args.input, format="jsonl"),
            train_target=DataLocation.local(str(out / "train.jsonl"), format="jsonl"),
            val_target=DataLocation.local(str(out / "val.jsonl"), format="jsonl"),
            stats_target=DataLocation.local(str(out / "dataset_stats.json"), format="json"),
            val_split=args.val_split,
            seed=args.seed,
        ))
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        raise SystemExit(1) from exc

    s = split.stats
    print(f"Total valid:   {s.total_valid_records}")
    print(f"Skipped:       {s.skipped_records}")
    print(f"Recipes:       {s.recipe_count}")
    print(f"Not-a-recipe:  {s.not_a_recipe_count}")
    print(f"Train:         {s.train_count}  → {out / 'train.jsonl'}")
    print(f"Val:           {s.val_count}  → {out / 'val.jsonl'}")
    print(f"Stats:         {out / 'dataset_stats.json'}")
