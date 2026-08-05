"""Composable dataset-to-export SFT pipeline."""

from __future__ import annotations

from dataclasses import dataclass

from labeling_sft.contracts import (
    ArtifactLocation,
    DatasetBuildRequest,
    DatasetSplit,
    ExportResult,
    TrainingResult,
)
from labeling_sft.interfaces import (
    BaseDatasetBuilder,
    BaseExporter,
    BaseTrainer,
)


@dataclass(frozen=True)
class PipelineResult:
    """Artifacts produced by one dataset-build, train, and export run."""

    dataset: DatasetSplit
    training: TrainingResult
    export: ExportResult


class SFTOrchestrator:
    """Run the SFT pipeline through interfaces, not concrete implementations.

    ``BaseTrainer`` owns data loading and preprocessing; this class passes its
    builder-produced ``DatasetSplit`` directly to the trainer.
    """

    def __init__(
        self,
        dataset_builder: BaseDatasetBuilder,
        trainer: BaseTrainer,
        exporter: BaseExporter,
    ) -> None:
        self._dataset_builder = dataset_builder
        self._trainer = trainer
        self._exporter = exporter

    def run(
        self,
        dataset_request: DatasetBuildRequest,
        training_target: ArtifactLocation,
        export_target: ArtifactLocation,
    ) -> PipelineResult:
        """Build, train, then export a model."""
        dataset = self._dataset_builder.build(dataset_request)
        return self.run_from_dataset(dataset, training_target, export_target)

    def run_from_dataset(
        self,
        dataset: DatasetSplit,
        training_target: ArtifactLocation,
        export_target: ArtifactLocation,
    ) -> PipelineResult:
        """Train and export from an already-built dataset split."""
        training = self._trainer.train(dataset, training_target)
        export = self._exporter.export(training, export_target)
        return PipelineResult(dataset=dataset, training=training, export=export)
