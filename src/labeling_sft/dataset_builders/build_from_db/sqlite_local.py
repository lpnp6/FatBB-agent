"""Build Alpaca train/validation JSONL files from a local SQLite database."""

from __future__ import annotations

import json
import random
import sqlite3
from pathlib import Path

from labeling_sft.contracts import (
    DataLocation,
    DataLocationType,
    DatasetBuildRequest,
    DatasetSplit,
)
from labeling_sft.interfaces.dataset_builder import BaseDatasetBuilder


class SqliteLocalBuilder(BaseDatasetBuilder):
    """Convert completed ``simhashes`` records to Alpaca-format splits."""

    def __init__(
        self,
        instruction: str = "Extract structured food knowledge from the following recipe.",
    ) -> None:
        self._instruction = instruction

    def build(self, request: DatasetBuildRequest) -> DatasetSplit:
        database_path = self._local_path(request.source, "source")
        if not database_path.is_file():
            raise FileNotFoundError(f"SQLite database not found: {database_path}")
        if not 0 <= request.val_split < 1:
            raise ValueError("val_split must be in [0.0, 1.0)")

        records = self._read(database_path)
        if not records:
            raise ValueError("no completed labeled records found in simhashes")

        random.Random(request.seed).shuffle(records)
        val_count = int(len(records) * request.val_split)
        val, train = records[:val_count], records[val_count:]
        self._write(self._local_path(request.train_target, "train_target"), train)
        self._write(self._local_path(request.val_target, "val_target"), val)

        return DatasetSplit(train=request.train_target, val=request.val_target)

    def _read(self, database_path: Path) -> list[dict[str, str]]:
        with sqlite3.connect(f"file:{database_path}?mode=ro", uri=True) as database:
            rows = database.execute(
                "SELECT raw_text, output FROM simhashes "
                "WHERE status = 'completed' "
                "AND raw_text IS NOT NULL AND output IS NOT NULL"
            )
            return [
                {"instruction": self._instruction, "input": raw_text, "output": output}
                for raw_text, output in rows
            ]

    @staticmethod
    def _write(path: Path, records: list[dict[str, str]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            for record in records:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")

    @staticmethod
    def _local_path(location: DataLocation, name: str) -> Path:
        if location.type is not DataLocationType.LOCAL_PATH:
            raise NotImplementedError(
                f"SqliteLocalBuilder only supports local paths; "
                f"{name} uses {location.type.value}"
            )
        return Path(location.value)
