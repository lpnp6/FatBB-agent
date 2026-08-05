"""QLoRA fine-tuning framework for Qwen2.5-3B-Instruct food knowledge extraction.

Package structure::

    interfaces/         Abstract base classes and data contracts
    configs/            Config implementations (QLoRAConfig, ...)
    dataset_builders/   DatasetBuilder implementations (Bootstrap, ...)
    trainers/           Trainer implementations (QLoRA, ...)
    exporters/          Exporter implementations (Merge, GGUF, ...)
    evaluators/         Evaluator implementations (Qwen, ...)
"""

# ── Interfaces (ABCs + contracts) ─────────────────────────────────────────
from labeling_sft.interfaces import (
    # Contracts
    ComparisonReport,
    DataLocation,
    DataLocationType,
    DatasetRecord,
    DatasetSplit,
    DatasetStats,
    EvalReport,
    ExportResult,
    TrainingResult,
    # ABCs
    BaseConfig,
    BaseDatasetBuilder,
    BaseEvaluator,
    BaseExporter,
    BaseTrainer,
)

# ── Concrete implementations ──────────────────────────────────────────────
from labeling_sft.configs import QLoRAConfig
from labeling_sft.dataset_builders import BootstrapDatasetBuilder
from labeling_sft.evaluators import QwenEvaluator
from labeling_sft.exporters import GGUFExporter, MergeExporter
from labeling_sft.trainers import QLoRATrainer

__all__ = [
    # Contracts
    "ComparisonReport",
    "DataLocation",
    "DataLocationType",
    "DatasetRecord",
    "DatasetSplit",
    "DatasetStats",
    "EvalReport",
    "ExportResult",
    "TrainingResult",
    # ABCs
    "BaseConfig",
    "BaseDatasetBuilder",
    "BaseEvaluator",
    "BaseExporter",
    "BaseTrainer",
    # Implementations
    "QLoRAConfig",
    "BootstrapDatasetBuilder",
    "QLoRATrainer",
    "MergeExporter",
    "GGUFExporter",
    "QwenEvaluator",
]
