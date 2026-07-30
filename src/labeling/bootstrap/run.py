"""Run the complete bootstrap workflow: sample once, then label/resume."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any

from ..clients import OpenAILabelingClient
from ..dedup.simhash_store import SimHashDedupStore
from ..prompts import RecipeLabelingPromptBuilder, RecipeRepairPromptBuilder
from .checkpoint import CheckpointManager
from .orchestrator import JsonlTrainingWriter, LabelingPipeline
from .sample_corpus import (
    assert_not_registered,
    build_manifests,
    persist_labeling_manifest,
    write_jsonl,
)


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True, help="Markdown corpus root")
    parser.add_argument("--source-name", default="corpus")
    parser.add_argument("--target", type=int, default=500)
    parser.add_argument("--holdout", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--threshold", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=Path("data/bootstrap"))
    parser.add_argument("--dedup-db", type=Path, default=Path("src/labeling/dedup/dedup_store.db"))
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-4o"))
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL"))
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def prepare_manifest(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    """Create the sampling artifacts once, or reuse them for a safe resume."""
    output_dir = args.output_dir.resolve()
    manifest_path = output_dir / "manifest.jsonl"
    if manifest_path.exists():
        logger.info("Reusing existing bootstrap manifest path=%s", manifest_path)
        return manifest_path, {"sampling": "reused"}

    source_name = args.source_name.strip()
    if not source_name:
        raise ValueError("--source-name must not be empty")
    if args.target < 1 or args.holdout < 0:
        raise ValueError("--target must be positive and --holdout must be non-negative")
    labeling_records, holdout_records, report = build_manifests(
        {source_name: args.source_dir.expanduser().resolve()}, {source_name: 1.0},
        args.target, args.holdout, args.seed, args.threshold,
    )
    dedup_db = args.dedup_db.resolve()
    assert_not_registered(labeling_records, dedup_db, args.threshold)
    write_jsonl(manifest_path, labeling_records)
    write_jsonl(output_dir / "holdout.jsonl", holdout_records)
    persist_labeling_manifest(labeling_records, dedup_db, args.threshold)
    report["dedup_store"] = str(dedup_db)
    report["registered_in_flight"] = len(labeling_records)
    (output_dir / "sampling_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    logger.info("Created bootstrap manifest count=%d path=%s", len(labeling_records), manifest_path)
    return manifest_path, {"sampling": "created", **report}


def main() -> None:
    args = parse_args()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY must be set")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(output_dir / "labeling.log", encoding="utf-8"),
        ],
    )
    try:
        manifest_path, summary = prepare_manifest(args)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    manifest = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    training_path = output_dir / "training.jsonl"
    dedup_store = SimHashDedupStore(args.dedup_db.resolve())
    try:
        pipeline = LabelingPipeline(
            client=OpenAILabelingClient(
                api_key=api_key, model=args.model,
                label_prompt_builder=RecipeLabelingPromptBuilder(),
                repair_prompt_builder=RecipeRepairPromptBuilder(),
                base_url=args.base_url, max_concurrent=args.concurrency, max_tokens=args.max_tokens,
            ),
            dedup_store=dedup_store,
            checkpoint=CheckpointManager(output_dir / "checkpoint.json", manifest_path=manifest_path, output_path=training_path),
            training_writer=JsonlTrainingWriter(training_path),
            retries=args.retries,
        )
        summary["labeling"] = asyncio.run(pipeline.run(manifest))
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    finally:
        dedup_store._db.close()


if __name__ == "__main__":
    main()
