"""Data contracts for model training.

Used by ``BaseTrainer`` → ``BaseExporter`` / ``BaseEvaluator``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrainingResult:
    """Return value of ``BaseTrainer.train()``.

    Exporter and Evaluator consume this contract to locate model artifacts —
    they never need to know how training was performed.
    """

    output_dir: str  # model / adapter save directory
    adapter_path: str  # LoRA adapter path (for full FT, same as output_dir)
    base_model_id: str  # HuggingFace base model ID
    final_eval_loss: float | None = None
    total_steps: int = 0
    best_checkpoint: str | None = None  # path to best checkpoint directory
