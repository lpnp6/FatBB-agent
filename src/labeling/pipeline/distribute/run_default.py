"""Run the default distributed labeling pipeline (Redis Streams + Ollama).

One entry point, two roles:

* ``--role orchestrator`` — discover, deduplicate, enqueue batches, and drain
  their results back through the result stream. Owns dedup/checkpoint state.
* ``--role worker`` — dequeue tasks, label via Ollama, publish results back.
  Stateless: only touches Redis + Ollama, never dedup/checkpoint.

Run one orchestrator and any number of workers (each with a distinct
``--consumer`` name).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any

from redis.asyncio import Redis

from ...checkpoint.file_store import FileCheckpointStore
from ...clients.ollama_client import OllamaLabelingClient
from ...dedup.simhash_store import SimHashDedupStore
from ...prompts import RecipeLabelingPromptBuilder, RecipeRepairPromptBuilder
from ...queue import RedisStreamsWorkQueue
from ...sampling.sampler import Sampler
from ...utils.uri_resolver import FileSystemURIResolver
from ...utils.validator import OutputValidator
from .orchestrator import DistributedProductionOrchestrator
from .worker import Worker

DEFAULT_DEDUP_DB = Path("src/labeling/dedup/dedup_store_bootstrap.sqlite")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--role", choices=("orchestrator", "worker"), default="orchestrator",
    )
    parser.add_argument("--base-dir", type=Path, default="data/markdown")
    parser.add_argument("--output-dir", type=Path, default="data/distributed")
    parser.add_argument("--model", default=os.environ.get("OLLAMA_MODEL", "qwen2.5-fatbb:v2-gpu"))
    parser.add_argument("--host", default=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"))
    parser.add_argument("--max-concurrent", type=int, default=1)
    parser.add_argument("--num-ctx", type=int, default=16384)
    parser.add_argument("--num-predict", type=int, default=9216)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--threshold", type=int, default=3)
    parser.add_argument("--count", type=int, default=1, help="Worker dequeue batch size.")
    parser.add_argument(
        "--no-repair", action="store_true",
        help="Skip repair on validation failure — just re-label instead.",
    )
    parser.add_argument(
        "--redis-url",
        default=os.environ.get("REDIS_URL", "redis://127.0.0.1:6379/0"),
    )
    parser.add_argument("--stream", default="labeling:tasks")
    parser.add_argument("--group", default="labeling-workers")
    parser.add_argument(
        "--consumer",
        default=os.environ.get("CONSUMER_NAME", f"worker-{os.getpid()}"),
    )
    return parser.parse_args()


def _build_queue(
    *,
    redis_url: str,
    stream: str,
    group: str,
    consumer: str,
) -> tuple[Redis, RedisStreamsWorkQueue]:
    """Create the Redis client and the transport queue (no stores — pure transport)."""
    client = Redis.from_url(redis_url)
    queue = RedisStreamsWorkQueue(
        client,
        consumer=consumer,
        stream=stream,
        group=group,
    )
    return client, queue


async def run_default(
    base_dir: Path | str,
    output_dir: Path,
    *,
    role: str,
    model: str,
    host: str,
    max_concurrent: int,
    num_ctx: int,
    num_predict: int,
    temperature: float,
    batch_size: int,
    threshold: int,
    count: int,
    repair: bool,
    redis_url: str,
    stream: str,
    group: str,
    consumer: str,
) -> dict[str, Any] | None:
    """Assemble and run one role of the distributed labeling pipeline."""
    output_dir.mkdir(parents=True, exist_ok=True)

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
        "Starting distributed pipeline role=%s base_dir=%s output_dir=%s "
        "stream=%s group=%s consumer=%s",
        role, base_dir, output_dir, stream, group, consumer,
    )

    source_dir = Path(base_dir).expanduser().resolve()
    client, queue = _build_queue(
        redis_url=redis_url, stream=stream, group=group, consumer=consumer,
    )
    dedup_store: SimHashDedupStore | None = None

    try:
        if role == "orchestrator":
            checkpoint = FileCheckpointStore(output_dir / "item_checkpoint.json")
            dedup_store = SimHashDedupStore(
                DEFAULT_DEDUP_DB.resolve(), threshold=threshold, checkpoint=checkpoint,
            )
            resolver = FileSystemURIResolver(base_dir=source_dir)
            sampler = Sampler(resolver, dedup_store, checkpoint, batch_size=batch_size)
            orchestrator = DistributedProductionOrchestrator(
                dedup_store=dedup_store,
                sampler=sampler,
                checkpoint=checkpoint,
                task_queue=queue,
            )
            return await orchestrator.run(source_dir)

        worker = Worker(
            queue=queue,
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
            validator=OutputValidator(mode="production"),
            repair=repair,
        )
        await worker.run(count=count)
        return None
    finally:
        if dedup_store is not None:
            dedup_store._db.close()
        await client.aclose()


def main() -> None:
    args = parse_args()
    result = asyncio.run(run_default(
        args.base_dir,
        args.output_dir,
        role=args.role,
        model=args.model,
        host=args.host,
        max_concurrent=args.max_concurrent,
        num_ctx=args.num_ctx,
        num_predict=args.num_predict,
        temperature=args.temperature,
        batch_size=args.batch_size,
        threshold=args.threshold,
        count=args.count,
        repair=not args.no_repair,
        redis_url=args.redis_url,
        stream=args.stream,
        group=args.group,
        consumer=args.consumer,
    ))
    if result is not None:
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
