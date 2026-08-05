"""Data contracts for labeling_sft.

Typed dataclasses for all cross-module communication —
ABCs and implementations import from here rather than
depending on each other's internal types.
"""

from labeling_sft.contracts.dataset import (
    DataLocation,
    DataLocationType,
    DatasetBuildRequest,
    DatasetRecord,
    DatasetSplit,
    DatasetStats,
)
from labeling_sft.contracts.evaluation import ComparisonReport, EvalReport
from labeling_sft.contracts.export import ExportResult
from labeling_sft.contracts.training import (
    ArtifactLocation,
    ArtifactLocationType,
    TrainingResult,
)

__all__ = [
    # Dataset
    "DataLocation",
    "DataLocationType",
    "DatasetBuildRequest",
    "DatasetRecord",
    "DatasetSplit",
    "DatasetStats",
    # Training
    "ArtifactLocation",
    "ArtifactLocationType",
    "TrainingResult",
    # Evaluation
    "EvalReport",
    "ComparisonReport",
    # Export
    "ExportResult",
]
