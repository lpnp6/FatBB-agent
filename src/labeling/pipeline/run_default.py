"""Run the default production labeling pipeline (local Ollama model).

Mirrors :mod:`labeling.bootstrap.run_default` but assembles a
:class:`ProductionOrchestrator` wired to a local Ollama backend with
production-mode (lenient) validation.  Every dependency is injected
through abstract interfaces so individual backends can be swapped
without changing pipeline code.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any

from ..checkpoint.file_store import FileCheckpointStore
from ..clients.ollama_client import OllamaLabelingClient
from ..dedup.simhash_store import SimHashDedupStore
from ..prompts import RecipeLabelingPromptBuilder, RecipeRepairPromptBuilder
from ..sampling.sampler import Sampler
from ..utils.uri_resolver import FileSystemURIResolver
from ..utils.validator import OutputValidator
from .orchestrator import ProductionOrchestrator

DEFAULT_DEDUP_DB = Path("src/labeling/dedup/dedup_store_bootstrap.sqlite")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-dir", type=Path, default="data/markdown",
    )
    parser.add_argument("--target", type=int, default=500)
    parser.add_argument("--output-dir", type=Path, default="data/production")
    parser.add_argument(
        "--model",
        default=os.environ.get("OLLAMA_MODEL", "qwen2.5-fatbb:v2-gpu"),
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"),
    )
    parser.add_argument("--max-concurrent", type=int, default=1)
    parser.add_argument("--num-ctx", type=int, default=16384)
    parser.add_argument("--num-predict", type=int, default=9216)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--threshold", type=int, default=3)
    parser.add_argument(
        "--no-repair", action="store_true",
        help="Skip repair on validation failure — just re-label instead.",
    )
    return parser.parse_args()


async def run_default(
    base_dir: Path | str,
    target: int,
    output_dir: Path,
    *,
    model: str,
    host: str,
    max_concurrent: int,
    num_ctx: int,
    num_predict: int,
    temperature: float,
    batch_size: int,
    retries: int,
    threshold: int,
    repair: bool = True,
) -> dict[str, Any]:
    """Assemble and run the standard local-files / Ollama labeling pipeline."""
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level="INFO",
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                output_dir / "labeling.log", encoding="utf-8",
            ),
        ],
    )

    logger = logging.getLogger(__name__)
    logger.info(
        "Starting production pipeline: base_dir=%s target=%d output_dir=%s "
        "model=%s host=%s max_concurrent=%d num_ctx=%d num_predict=%d "
        "batch_size=%d retries=%d threshold=%d repair=%s",
        base_dir, target, output_dir,
        model, host, max_concurrent, num_ctx, num_predict,
        batch_size, retries, threshold, repair,
    )

    source_dir = Path(base_dir).expanduser().resolve()
    checkpoint = FileCheckpointStore(output_dir / "item_checkpoint.json")
    dedup_store = SimHashDedupStore(
        DEFAULT_DEDUP_DB.resolve(),
        threshold=threshold,
        checkpoint=checkpoint,
    )
    try:
        resolver = FileSystemURIResolver(base_dir=source_dir)
        sampler = Sampler(
            resolver, dedup_store, checkpoint, batch_size=batch_size,
        )
        orchestrator = ProductionOrchestrator(
            client=OllamaLabelingClient(
                model=model,
                host=host,
                label_prompt_builder=RecipeLabelingPromptBuilder(),
                repair_prompt_builder=RecipeRepairPromptBuilder(),
                max_concurrent=max_concurrent,
                num_ctx=num_ctx,
                num_predict=num_predict,
                temperature=temperature,
            ),
            dedup_store=dedup_store,
            sampler=sampler,
            checkpoint=checkpoint,
            retries=retries,
            repair=repair,
            validator=OutputValidator(mode="production"),
        )
        return await orchestrator.run(source_dir, target)
    finally:
        dedup_store._db.close()


def main() -> None:
    args = parse_args()
    result = asyncio.run(run_default(
        args.base_dir,
        args.target,
        args.output_dir,
        model=args.model,
        host=args.host,
        max_concurrent=args.max_concurrent,
        num_ctx=args.num_ctx,
        num_predict=args.num_predict,
        temperature=args.temperature,
        batch_size=args.batch_size,
        retries=args.retries,
        threshold=args.threshold,
        repair=not args.no_repair,
    ))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
