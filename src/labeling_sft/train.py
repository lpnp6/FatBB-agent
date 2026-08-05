"""Backward-compatible re-export from the new ``trainers/`` sub-package.

.. deprecated:: transitional
    Import from :mod:`labeling_sft.trainers` directly in new code.
    Use :class:`QLoRATrainer` instead of the bare ``run_training()`` function.
"""

from __future__ import annotations

import logging

# Re-export key symbols so existing imports don't break
from labeling_sft.trainers.qlora import (
    QLoRATrainer,
    _CompletionOnlyCollator,
    _gpu_snapshot,
    _make_memory_watchdog,
    format_example,
    load_system_prompt,
)
from labeling_sft.configs.qlora import QLoRAConfig
from labeling_sft.contracts import ArtifactLocation, DataLocation, DatasetSplit, TrainingResult

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


def run_training(config: QLoRAConfig) -> TrainingResult:
    """Execute the full QLoRA training pipeline.

    .. deprecated::
        Use :meth:`QLoRATrainer.train()` instead, which returns a typed
        :class:`~labeling_sft.interfaces.contracts.TrainingResult`.
    """
    trainer = QLoRATrainer(config)
    return trainer.train(DatasetSplit(
        train=DataLocation.local(str(config.project_dir / "Alpaca" / "train.jsonl"), format="jsonl"),
        val=DataLocation.local(str(config.project_dir / "Alpaca" / "val.jsonl"), format="jsonl"),
    ), ArtifactLocation.local(str(config.project_dir)))
