"""Build Alpaca train/validation JSONL files from a local directory."""

from __future__ import annotations

import json
import random
from pathlib import Path

from labeling_sft.contracts import (
    DataLocation,
    DataLocationType,
    DatasetBuildRequest,
    DatasetSplit,
)
from labeling_sft.interfaces.dataset_builder import BaseDatasetBuilder


class BuildFromFileDatasetBuilder(BaseDatasetBuilder):
    """Convert all ``*.jsonl`` files in a directory to Alpaca-format splits."""

    def __init__(
        self,
        instruction: str = "Extract structured food knowledge from the following recipe.",
    ) -> None:
        self._instruction = instruction

    def build(self, request: DatasetBuildRequest) -> DatasetSplit:
        source_dir = self._local_path(request.source, "source")
        if not source_dir.is_dir():
            raise NotADirectoryError(f"Source directory not found: {source_dir}")
        if not 0 <= request.val_split < 1:
            raise ValueError("val_split must be in [0.0, 1.0)")

        source_files = sorted(source_dir.glob("*.jsonl"))
        if not source_files:
            raise FileNotFoundError(f"No JSONL files found in: {source_dir}")

        records = [record for path in source_files for record in self._read(path)]
        if not records:
            raise ValueError("no records found in source directory")

        random.Random(request.seed).shuffle(records)
        val_count = int(len(records) * request.val_split)
        val, train = records[:val_count], records[val_count:]
        train_path, val_path = self._output_paths(request.project_name, request.work_dir)
        self._write(train_path, train)
        self._write(val_path, val)

        return DatasetSplit(
            train=DataLocation.local(str(train_path), format="jsonl"),
            val=DataLocation.local(str(val_path), format="jsonl"),
        )

    def _read(self, path: Path) -> list[dict[str, str]]:
        with path.open(encoding="utf-8") as file:
            return [self._to_alpaca(json.loads(line)) for line in file if line.strip()]

    def _to_alpaca(self, record: dict[str, object]) -> dict[str, str]:
        output = record["output"]
        return {
            "instruction": self._instruction,
            "input": str(record["input"]),
            "output": output if isinstance(output, str) else json.dumps(
                output, ensure_ascii=False, separators=(",", ":")
            ),
        }

    @staticmethod
    def _write(path: Path, records: list[dict[str, str]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            for record in records:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")

    @staticmethod
    def _output_paths(project_name: str, work_dir: str | None) -> tuple[Path, Path]:
        output_dir = (Path(work_dir).expanduser() if work_dir else Path.home() / ".fatbb" / project_name) / "Alpaca"
        return output_dir / "train.jsonl", output_dir / "val.jsonl"

    @staticmethod
    def _local_path(location: DataLocation, name: str) -> Path:
        if location.type is not DataLocationType.LOCAL_PATH:
            raise NotImplementedError(
                f"BuildFromFileDatasetBuilder only supports local paths; "
                f"{name} uses {location.type.value}"
            )
        return Path(location.value)
