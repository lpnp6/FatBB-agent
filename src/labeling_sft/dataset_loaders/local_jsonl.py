"""Local JSONL dataset loader."""

from __future__ import annotations

import json
from pathlib import Path

from labeling_sft.contracts import DataLocation, DataLocationType, DatasetRecord
from labeling_sft.interfaces.dataset_loader import BaseDatasetLoader


class LocalJsonlDatasetLoader(BaseDatasetLoader):
    """Load Alpaca records from a local JSONL file."""

    def load(self, location: DataLocation) -> list[DatasetRecord]:
        if location.type is not DataLocationType.LOCAL_PATH:
            raise NotImplementedError(
                f"LocalJsonlDatasetLoader only supports local paths; "
                f"location uses {location.type.value}"
            )

        path = Path(location.value)
        if not path.is_file():
            raise FileNotFoundError(f"Dataset file not found: {path}")
        if path.stat().st_size == 0:
            raise ValueError(f"Dataset file is empty: {path}")

        with path.open(encoding="utf-8") as file:
            return [DatasetRecord(**json.loads(line)) for line in file if line.strip()]
