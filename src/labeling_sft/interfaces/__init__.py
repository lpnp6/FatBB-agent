"""Abstract interfaces and data contracts for labeling_sft.

All cross-module communication is typed through the dataclasses in
:mod:`~labeling_sft.interfaces.contracts`.  Concrete implementations live
in the sibling sub-packages (``configs/``, ``trainers/``, etc.).
"""

from labeling_sft.contracts import (
    ComparisonReport,
    DataLocation,
    DataLocationType,
    DatasetBuildRequest,
    DatasetRecord,
    DatasetSplit,
    DatasetStats,
    EvalReport,
    ExportResult,
    TrainingResult,
)
from labeling_sft.interfaces.config import BaseConfig
from labeling_sft.interfaces.dataset_builder import BaseDatasetBuilder
from labeling_sft.interfaces.evaluator import BaseEvaluator
from labeling_sft.interfaces.exporter import BaseExporter
from labeling_sft.interfaces.trainer import BaseTrainer

__all__ = [
    # Contracts
    "ComparisonReport",
    "DataLocation",
    "DataLocationType",
    "DatasetBuildRequest",
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
]
