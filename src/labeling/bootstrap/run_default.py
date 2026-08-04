"""Run the default bootstrap labeling pipeline."""

from __future__ import annotations

import argparse
import asyncio
from email.mime import base
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

DEFAULT_DEDUP_DB = Path("src/labeling/dedup/dedup_store_bootstrap.sqlite")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default="data/markdown")
    parser.add_argument("--target", type=int, default=500)
    parser.add_argument("--output-dir", type=Path, default="data/bootstrap")
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "deepseek-v4-flash"))
    parser.add_argument("--base-url", default=os.environ.get("BASE_URL", "https://api.deepseek.com"))
    parser.add_argument("--max-concurrent", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--threshold", type=int, default=3)
    return parser.parse_args()


async def run_default(
    base_dir: Path | str,
    target: int,
    output_dir: Path,
    *,
    model: str,
    base_url: str,
    max_concurrent: int,
    max_tokens: int,
    batch_size: int,
    retries: int,
    threshold: int,
) -> dict[str, Any]:
    """Assemble and run the standard local-files/OpenAI labeling pipeline."""
    output_dir.mkdir(parents=True, exist_ok=True)
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY must be set")

    logging.basicConfig(
        level="INFO",
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(output_dir / "labeling.log", encoding="utf-8"),
        ],
    )

    logger = logging.getLogger(__name__)
    logger.info(
        "Starting bootstrap pipeline: base_dir=%s target=%d output_dir=%s "
        "model=%s base_url=%s max_concurrent=%d max_tokens=%d "
        "batch_size=%d retries=%d threshold=%d",
        base_dir, target, output_dir,
        model, base_url, max_concurrent, max_tokens, batch_size, retries, threshold,
    )

    source_dir = Path(base_dir).expanduser().resolve()
    checkpoint = FileCheckpointStore(output_dir / "item_checkpoint.json")
    dedup_store = SimHashDedupStore(
        DEFAULT_DEDUP_DB.resolve(), threshold=threshold, checkpoint=checkpoint,
    )
    try:
        resolver = FileSystemURIResolver(base_dir=source_dir)
        sampler = Sampler(resolver, dedup_store, checkpoint, batch_size=batch_size)
        orchestrator = BootstrapOrchestrator(
            client=OpenAILabelingClient(
                api_key=api_key,
                model=model,
                label_prompt_builder=RecipeLabelingPromptBuilder(),
                repair_prompt_builder=RecipeRepairPromptBuilder(),
                base_url=base_url,
                max_concurrent=max_concurrent,
                max_tokens=max_tokens,
            ),
            dedup_store=dedup_store,
            sampler=sampler,
            checkpoint=checkpoint,
            retries=retries,
        )
        return await orchestrator.run(source_dir, target)
    finally:
        dedup_store._db.close()


def main() -> None:
    args = parse_args()
    result = asyncio.run(run_default(
        args.base_dir, args.target, args.output_dir,
        model=args.model,
        base_url=args.base_url,
        max_concurrent=args.max_concurrent,
        max_tokens=args.max_tokens,
        batch_size=args.batch_size,
        retries=args.retries,
        threshold=args.threshold,
    ))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
