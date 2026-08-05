"""Backward-compatible re-export from the new ``trainers/`` sub-package.

.. deprecated:: transitional
    Import from :mod:`labeling_sft.trainers` directly in new code.
    Use :class:`QLoRATrainer` instead of the bare ``run_training()`` function.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
# Re-export key symbols so existing imports don't break
from labeling_sft.trainers.qlora import (
    QLoRATrainer,
    _CompletionOnlyCollator,
    _gpu_snapshot,
    _make_memory_watchdog,
    format_example,
    load_system_prompt,
)
from labeling_sft.configs.qlora import QLoRAConfig, _qlora_config_fields as _config_fields  # noqa: F401
from labeling_sft.contracts import DataLocation, DatasetSplit

__all__ = [
    "QLoRATrainer",
    "QLoRAConfig",
    "_CompletionOnlyCollator",
    "_gpu_snapshot",
    "_make_memory_watchdog",
    "format_example",
    "load_system_prompt",
    "run_training",
]

logger = logging.getLogger(__name__)


def run_training(config: QLoRAConfig) -> None:
    """Execute the full QLoRA training pipeline.

    .. deprecated::
        Use :meth:`QLoRATrainer.train()` instead, which returns a typed
        :class:`~labeling_sft.interfaces.contracts.TrainingResult`.
    """
    trainer = QLoRATrainer(config)
    trainer.train(DatasetSplit(
        train=DataLocation.local(str(Path(config.output_dir) / "train.jsonl"), format="jsonl"),
        val=DataLocation.local(str(Path(config.output_dir) / "val.jsonl"), format="jsonl"),
    ))


# CLI entry point (delegates to QLoRATrainer's CLI)
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    from labeling_sft.trainers.qlora import _build_arg_parser

    parser = _build_arg_parser()
    args = parser.parse_args()
    config = QLoRAConfig.from_cli_args(args)
    run_training(config)
