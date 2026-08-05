"""Abstract base class for model trainers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .config import BaseConfig
from labeling_sft.contracts import ArtifactLocation, DatasetSplit, TrainingResult


class BaseTrainer(ABC):
    """Train a model using a :class:`BaseConfig` and a dataset split.

    The training pipeline is split into three independently-overridable stages:

    1. :meth:`load_data`   — data loading and preprocessing
    2. :meth:`load_model`  — model loading and adapter setup
    3. :meth:`train`       — training loop

    Subclasses can replace just one stage (e.g. swap model architecture)
    while inheriting the other two.

    Concrete implementations:
        - :class:`~labeling_sft.trainers.qlora.QLoRATrainer`
        - Future: ``FullFinetuneTrainer``, ``DPOTrainer``, ``DistributedTrainer``, ...
    """

    def __init__(self, config: BaseConfig) -> None:
        self.config = config

    @abstractmethod
    def load_data(
        self,
        split: DatasetSplit,
    ) -> tuple[Any, Any]:
        """Load and preprocess training / validation data.

        Returns:
            ``(train_dataset, val_dataset)`` — tokenized HuggingFace Datasets.
        """
        ...

    @abstractmethod
    def load_model(self) -> tuple[Any, Any]:
        """Load and configure the model and tokenizer.

        Returns:
            ``(model, tokenizer)`` — model on device, with LoRA applied if applicable.
        """
        ...

    @abstractmethod
    def train(
        self,
        split: DatasetSplit,
        artifact_target: ArtifactLocation,
    ) -> TrainingResult:
        """Execute the full training pipeline.

        Default implementation pattern::

            train_ds, val_ds = self.load_data(split)
            model, tokenizer = self.load_model()
            # ... training loop ...
            return TrainingResult(...)

        Args:
            split: Locations and optional statistics produced by a dataset builder.
            artifact_target: Destination for the trained model artifact.

        Returns:
            :class:`~labeling_sft.interfaces.contracts.TrainingResult`.
        """
        ...
