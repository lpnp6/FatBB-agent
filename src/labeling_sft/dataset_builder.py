"""Backward-compatible re-export from the new ``dataset_builders/`` sub-package.

.. deprecated:: transitional
    Import from :mod:`labeling_sft.dataset_builders` directly in new code.
    Use :func:`BuildFromFileDatasetBuilder.build()` instead of the bare
    ``build_dataset()`` function.
"""

from __future__ import annotations

from pathlib import Path

from labeling_sft.contracts import DataLocation, DatasetBuildRequest, DatasetSplit
from labeling_sft.dataset_builders.build_from_file import BuildFromFileDatasetBuilder

# Re-export the class
__all__ = ["BuildFromFileDatasetBuilder", "build_dataset"]


def build_dataset(
    input_dir: str,
    train_path: str = "data/training/train.jsonl",
    val_path: str = "data/training/val.jsonl",
    val_split: float = 0.15,
    seed: int = 42,
) -> DatasetSplit:
    """Convert directory JSONL files to Alpaca-format train/val splits.

    .. deprecated::
        Use :meth:`BuildFromFileDatasetBuilder.build()` instead, which returns
        a typed :class:`~labeling_sft.interfaces.contracts.DatasetSplit`.
    """
    builder = BuildFromFileDatasetBuilder()
    split = builder.build(DatasetBuildRequest(
        source=DataLocation.local(input_dir, format="jsonl"),
        train_target=DataLocation.local(train_path, format="jsonl"),
        val_target=DataLocation.local(val_path, format="jsonl"),
        val_split=val_split,
        seed=seed,
    ))
    return split


# CLI entry point
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert directory JSONL files to Alpaca-format train/val splits"
    )
    parser.add_argument(
        "--input-dir", required=True,
        help="Directory containing labeled JSONL files",
    )
    parser.add_argument(
        "--output_dir", default="models/qwen2.5-3b-fatbb-v1",
        help="Directory to write train.jsonl and val.jsonl",
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
        split = build_dataset(
            input_dir=args.input_dir,
            train_path=str(out / "train.jsonl"),
            val_path=str(out / "val.jsonl"),
            val_split=args.val_split,
            seed=args.seed,
        )
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        raise SystemExit(1) from exc

    print(f"Train: {split.train.value}")
    print(f"Val:   {split.val.value}")
