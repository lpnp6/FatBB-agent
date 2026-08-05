"""Abstract base class for dataset builders."""

from __future__ import annotations

from abc import ABC, abstractmethod

from labeling_sft.contracts import DatasetSplit


class BaseDatasetBuilder(ABC):
    """Convert raw labeled data into Alpaca-format train/val splits.

    Trainer and Evaluator only consume the produced JSONL files;
    they are intentionally unaware of the data source.

    Concrete implementations:
        - :class:`~labeling_sft.dataset_builders.bootstrap.BootstrapDatasetBuilder`
        - Future: ``SyntheticDatasetBuilder``, ``MultiSourceDatasetBuilder``, ...
    """

    @abstractmethod
    def build(
        self,
        input_path: str,
        train_path: str,
        val_path: str,
        stats_path: str,
        val_split: float = 0.15,
        seed: int = 42,
    ) -> DatasetSplit:
        """Execute dataset build.

        Args:
            input_path: Path to raw labeled data (format depends on subclass).
            train_path: Where to write training split.
            val_path:   Where to write validation split.
            stats_path: Where to write dataset statistics JSON.
            val_split:  Validation fraction (0.0–1.0).
            seed:       Random seed.

        Returns:
            DatasetSplit with output paths and statistics.

        Raises:
            FileNotFoundError: ``input_path`` does not exist.
            ValueError:        No valid records found.
        """
        ...

    @abstractmethod
    def load_split(
        self,
        train_path: str,
        val_path: str,
    ) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
        """Load a pre-built split back into memory as record lists.

        Each record is ``{"instruction": ..., "input": ..., "output": ...}``.

        Used by :meth:`BaseTrainer.load_data` and by Evaluator to read data.

        Returns:
            ``(train_records, val_records)``
        """
        ...
