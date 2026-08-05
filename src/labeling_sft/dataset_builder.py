"""Backward-compatible re-export from the new ``dataset_builders/`` sub-package.

.. deprecated:: transitional
    Import from :mod:`labeling_sft.dataset_builders` directly in new code.
    Use :func:`BootstrapDatasetBuilder.build()` instead of the bare
    ``build_dataset()`` function.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from labeling_sft.dataset_builders.bootstrap import BootstrapDatasetBuilder

# Re-export the class
__all__ = ["BootstrapDatasetBuilder", "build_dataset"]


def build_dataset(
    input_path: str = "data/bootstrap/training.jsonl",
    train_path: str = "data/training/train.jsonl",
    val_path: str = "data/training/val.jsonl",
    stats_path: str = "data/training/dataset_stats.json",
    val_split: float = 0.15,
    seed: int = 42,
) -> dict[str, Any]:
    """Convert bootstrap JSONL to Alpaca-format train/val splits.

    .. deprecated::
        Use :meth:`BootstrapDatasetBuilder.build()` instead, which returns
        a typed :class:`~labeling_sft.interfaces.contracts.DatasetSplit`.
    """
    builder = BootstrapDatasetBuilder()
    split = builder.build(
        input_path=input_path,
        train_path=train_path,
        val_path=val_path,
        stats_path=stats_path,
        val_split=val_split,
        seed=seed,
    )
    s = split.stats
    return {
        "total_valid_records": s.total_valid_records,
        "skipped_records": s.skipped_records,
        "recipe_count": s.recipe_count,
        "not_a_recipe_count": s.not_a_recipe_count,
        "train_count": s.train_count,
        "val_count": s.val_count,
        "val_split": s.val_split,
        "seed": s.seed,
        "train_domains": {d: v["train"] for d, v in s.domain_distribution.items()},
        "val_domains": {d: v["val"] for d, v in s.domain_distribution.items()},
    }


# CLI entry point
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
