"""Abstract base class for dataset builders."""

from __future__ import annotations

from abc import ABC, abstractmethod

from labeling_sft.contracts import DatasetBuildRequest, DatasetSplit


class BaseDatasetBuilder(ABC):
    """Convert raw labeled data into Alpaca-format train/val splits.

    Trainer and Evaluator only consume the produced JSONL files;
    they are intentionally unaware of the data source.

    Concrete implementations:
        - :class:`~labeling_sft.dataset_builders.build_from_file.BuildFromFileDatasetBuilder`
    """

    @abstractmethod
    def build(
        self,
        request: DatasetBuildRequest,
    ) -> DatasetSplit:
        """Execute dataset build.

        Args:
            request: Source location, artifact targets, and split parameters.

        Returns:
            DatasetSplit with output locations and statistics.

        Raises:
            FileNotFoundError: The source does not exist.
            ValueError:        No valid records found.
        """
        ...
