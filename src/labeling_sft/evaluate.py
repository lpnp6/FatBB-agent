"""Backward-compatible re-export from the new ``evaluators/`` sub-package.

.. deprecated:: transitional
    Import from :mod:`labeling_sft.evaluators` directly in new code.
    Use :class:`QwenEvaluator` instead of the bare ``evaluate()`` /
    ``evaluate_with_comparison()`` functions.
"""

from __future__ import annotations

from typing import Any

from labeling_sft.evaluators.qwen import (
    QwenEvaluator,
    _check_enum_values,
    _extract_json,
    _per_field_coverages,
)

__all__ = [
    "QwenEvaluator",
    "_extract_json",
    "_check_enum_values",
    "_per_field_coverages",
    "evaluate",
    "evaluate_with_comparison",
]


def evaluate(
    adapter_dir: str,
    val_file: str = "data/training/val.jsonl",
    base_model_id: str = "Qwen/Qwen2.5-3B-Instruct",
    output_report: str | None = "data/training/eval_report.json",
    *,
    local_files_only: bool = False,
    max_samples: int | None = None,
) -> dict[str, Any]:
    """Evaluate the fine-tuned model only.

    .. deprecated::
        Use :meth:`QwenEvaluator.evaluate()` instead, which returns a typed
        :class:`~labeling_sft.interfaces.contracts.EvalReport`.
    """
    evaluator = QwenEvaluator(base_model_id=base_model_id)
    report = evaluator.evaluate(
        adapter_dir=adapter_dir,
        val_path=val_file,
        base_model_id=base_model_id,
        output_report=output_report,
        max_samples=max_samples,
        local_files_only=local_files_only,
    )
    return report.raw_metrics


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

    .. deprecated::
        Use :meth:`QwenEvaluator.compare()` instead, which returns a typed
        :class:`~labeling_sft.interfaces.contracts.ComparisonReport`.
    """
    evaluator = QwenEvaluator(base_model_id=base_model_id)
    report = evaluator.compare(
        adapter_dir=adapter_dir,
        val_path=val_file,
        base_model_id=base_model_id,
        output_report=output_report,
        diff_examples=diff_examples,
        max_samples=max_samples,
        local_files_only=local_files_only,
    )
    return {
        "base_model_id": report.base_model_id,
        "adapter_dir": report.adapter_dir,
        "base": report.base.raw_metrics,
        "fine_tuned": report.fine_tuned.raw_metrics,
    }


# CLI entry point
if __name__ == "__main__":
    import argparse
    import logging

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
