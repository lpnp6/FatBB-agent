"""Abstract dataset location resolver."""

from __future__ import annotations

from abc import ABC, abstractmethod

from labeling_sft.contracts import DataLocation, DatasetRecord


class BaseDatasetLoader(ABC):
    """Load standardized dataset records from a data location."""

    @abstractmethod
    def load(self, location: DataLocation) -> list[DatasetRecord]:
        ...
