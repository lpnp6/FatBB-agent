"""Run the default SQLite → QLoRA → GGUF SFT pipeline."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from labeling_sft.configs import QLoRAConfig
from labeling_sft.contracts import ArtifactLocation, DataLocation, DatasetBuildRequest, DatasetSplit
from labeling_sft.dataset_builders import SqliteLocalBuilder
from labeling_sft.dataset_loaders import LocalJsonlDatasetLoader
from labeling_sft.exporters import GGUFExporter
from labeling_sft.orchestrator import PipelineResult, SFTOrchestrator
from labeling_sft.trainers import QLoRATrainer


def load_config(source: str | Path | Mapping[str, Any]) -> QLoRAConfig:
    """Read and validate a JSON config mapping for the default pipeline."""
    data = (
        json.loads(Path(source).read_text(encoding="utf-8"))
        if isinstance(source, (str, Path)) else dict(source)
    )
    config = QLoRAConfig.from_dict(data)
    if problems := config.validate():
        raise ValueError("Invalid QLoRA config: " + "; ".join(problems))
    return config


def _resume_dataset(work_dir: Path) -> DatasetSplit | None:
    """Return the persisted split when a previous training checkpoint exists."""
    if not any(path.is_dir() for path in work_dir.glob("checkpoint-*")):
        return None

    train = work_dir / "Alpaca" / "train.jsonl"
    val = work_dir / "Alpaca" / "val.jsonl"
    if not train.is_file() or not val.is_file() or not train.stat().st_size or not val.stat().st_size:
        raise RuntimeError(
            "Cannot resume: a training checkpoint exists but the persisted "
            "Alpaca train/val split is missing or empty."
        )
    return DatasetSplit(
        train=DataLocation.local(str(train), format="jsonl"),
        val=DataLocation.local(str(val), format="jsonl"),
    )


def run_default(
    config: str | Path | Mapping[str, Any],
    database: str | Path,
    work_dir: str | Path,
    *,
    outtype: str = "q8_0",
) -> PipelineResult:
    """Assemble concrete local implementations and run the default pipeline."""
    output = Path(work_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    training_config = replace(load_config(config), work_dir=str(output))
    pipeline = SFTOrchestrator(
        SqliteLocalBuilder(),
        QLoRATrainer(training_config, LocalJsonlDatasetLoader()),
        GGUFExporter(outtype),
    )
    training_target = ArtifactLocation.local(str(output))
    export_target = ArtifactLocation.local(str(output / f"model-{outtype}.gguf"))
    if dataset := _resume_dataset(output):
        return pipeline.run_from_dataset(dataset, training_target, export_target)
    return pipeline.run(
        DatasetBuildRequest(
            source=DataLocation.local(str(Path(database).expanduser().resolve())),
            project_name=training_config.project_name,
            val_split=training_config.val_split,
            seed=training_config.seed,
            work_dir=str(output),
        ),
        training_target,
        export_target,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="QLoRA JSON config")
    parser.add_argument("--database", required=True, type=Path, help="Local SQLite labeling database")
    parser.add_argument("--work-dir", required=True, type=Path, help="Output directory")
    parser.add_argument("--outtype", default="q8_0", help="GGUF quantization type")
    args = parser.parse_args()
    result = run_default(args.config, args.database, args.work_dir, outtype=args.outtype)
    print(json.dumps({"model": result.training.model.value, "export": result.export.artifact.value}))


if __name__ == "__main__":
    main()
