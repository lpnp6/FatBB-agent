"""Run the default bootstrap labeling pipeline."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any

from ..checkpoint.file_store import FileCheckpointStore
from ..clients.openai_client import OpenAILabelingClient
from ..dedup.simhash_store import SimHashDedupStore
from ..prompts import RecipeLabelingPromptBuilder, RecipeRepairPromptBuilder
from ..sampling.sampler import Sampler
from ..utils.uri_resolver import FileSystemURIResolver
from .orchestrator import BootstrapOrchestrator

DEFAULT_DEDUP_DB = Path("src/labeling/dedup/dedup_store.sqlite")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default="data/markdown")
    parser.add_argument("--target", type=int, default=500)
    parser.add_argument("--output-dir", type=Path, default="data/bootstrap")
    return parser.parse_args()


async def run_default(
    base_dir: Path | str, target: int, output_dir: Path
) -> dict[str, Any]:
    """Assemble and run the standard local-files/OpenAI labeling pipeline."""
    output_dir.mkdir(parents=True, exist_ok=True)
    base_url = os.environ.get("BASE_URL", "https://api.deepseek.com")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY must be set")

    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level="INFO",
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(output_dir / "labeling.log", encoding="utf-8"),
        ],
    )

    source_dir = Path(base_dir).expanduser().resolve()
    checkpoint = FileCheckpointStore(output_dir / "item_checkpoint.json")
    dedup_store = SimHashDedupStore(DEFAULT_DEDUP_DB.resolve(), checkpoint=checkpoint)
    try:
        resolver = FileSystemURIResolver(base_dir=source_dir)
        sampler = Sampler(resolver, dedup_store, checkpoint)
        orchestrator = BootstrapOrchestrator(
            client=OpenAILabelingClient(
                api_key=api_key,
                model=os.environ.get("OPENAI_MODEL", "deepseek-v4-flash"),
                label_prompt_builder=RecipeLabelingPromptBuilder(),
                repair_prompt_builder=RecipeRepairPromptBuilder(),
                base_url=os.environ.get("OPENAI_BASE_URL"),
            ),
            dedup_store=dedup_store,
            sampler=sampler,
            checkpoint=checkpoint,
        )
        return await orchestrator.run(source_dir, target)
    finally:
        dedup_store._db.close()


def main() -> None:
    args = parse_args()
    result = asyncio.run(run_default(args.base_dir, args.target, args.output_dir))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
