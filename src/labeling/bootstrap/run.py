"""Run the complete bootstrap workflow: discover, dedup, label, persist.

Resume is built-in: re-run with the same ``--output-dir`` and the orchestrator
picks up where it left off (checkpoint + dedup store are the source of truth).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path

from ..clients import OpenAILabelingClient
from ..checkpoint.file_store import FileCheckpointStore
from ..dedup.simhash_store import SimHashDedupStore
from ..prompts import RecipeLabelingPromptBuilder, RecipeRepairPromptBuilder
from ..sampling.sampler import Sampler
from ..utils.uri_resolver import FileSystemURIResolver
from .orchestrator import BootstrapOrchestrator


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True, help="Markdown corpus root")
    parser.add_argument("--source-name", default="corpus")
    parser.add_argument("--target", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--threshold", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=Path("data/bootstrap"))
    parser.add_argument("--dedup-db", type=Path, default=Path("src/labeling/dedup/dedup_store.sqlite"))
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-4o"))
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL"))
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


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

    source_dir = args.source_dir.expanduser().resolve()
    checkpoint = FileCheckpointStore(output_dir / "item_checkpoint.json")
    dedup_store = SimHashDedupStore(args.dedup_db.resolve(), checkpoint=checkpoint)

    try:
        resolver = FileSystemURIResolver(base_dir=source_dir)
        sampler = Sampler(resolver, dedup_store, checkpoint)

        orchestrator = BootstrapOrchestrator(
            client=OpenAILabelingClient(
                api_key=api_key, model=args.model,
                label_prompt_builder=RecipeLabelingPromptBuilder(),
                repair_prompt_builder=RecipeRepairPromptBuilder(),
                base_url=args.base_url, max_concurrent=args.concurrency,
                max_tokens=args.max_tokens,
            ),
            dedup_store=dedup_store,
            sampler=sampler,
            checkpoint=checkpoint,
            retries=args.retries,
        )

        result = asyncio.run(orchestrator.run(
            source_dir, args.target,
        ))
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    finally:
        dedup_store._db.close()


if __name__ == "__main__":
    main()
