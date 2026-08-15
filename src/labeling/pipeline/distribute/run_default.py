"""Run the default distributed labeling pipeline (Redis Streams + Ollama).

One entry point, two roles:

* ``--role orchestrator`` — discover, deduplicate, enqueue batches, and drain
  their results back through the result stream. Owns dedup/checkpoint state.
* ``--role worker`` — dequeue tasks, label via Ollama, publish results back.
  Stateless: only touches Redis + Ollama, never dedup/checkpoint.

Run one orchestrator and any number of workers (each with a distinct
``--consumer`` name).

Config is read from ``.env.orchestrator`` / ``.env.worker`` (selected by
``--role``, override with ``--env-file``).  CLI flags take precedence over the
config file.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
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


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw and raw.strip() else default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return float(raw) if raw and raw.strip() else default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if not raw or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def parse_args() -> argparse.Namespace:
    # Pre-scan --role / --env-file so the right config file is loaded before
    # the real argument defaults are resolved (config selection precedes defaults).
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--role", choices=("orchestrator", "worker"))
    pre.add_argument("--env-file", type=Path, default=None)
    pre_ns, _ = pre.parse_known_args()

    role = pre_ns.role or os.environ.get("ROLE") or "orchestrator"
    env_file = pre_ns.env_file or Path(f".env.{role}")
    load_dotenv(env_file)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--role", choices=("orchestrator", "worker"),
        default=os.environ.get("ROLE", "orchestrator"),
    )
    parser.add_argument(
        "--env-file", type=Path, default=env_file,
        help="Config file (KEY=VALUE). Defaults to .env.<role>.",
    )
    parser.add_argument(
        "--base-dir", type=Path,
        default=os.environ.get("BASE_DIR", "data/markdown"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=os.environ.get("OUTPUT_DIR", "data/distributed"),
    )
    parser.add_argument("--model", default=os.environ.get("OLLAMA_MODEL", "qwen2.5-fatbb:v2-gpu"))
    parser.add_argument("--host", default=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"))
    parser.add_argument("--max-concurrent", type=int, default=_env_int("MAX_CONCURRENT", 1))
    parser.add_argument("--num-ctx", type=int, default=_env_int("NUM_CTX", 16384))
    parser.add_argument("--num-predict", type=int, default=_env_int("NUM_PREDICT", 9216))
    parser.add_argument("--temperature", type=float, default=_env_float("TEMPERATURE", 0.1))
    parser.add_argument("--batch-size", type=int, default=_env_int("BATCH_SIZE", 1))
    parser.add_argument("--threshold", type=int, default=_env_int("THRESHOLD", 3))
    parser.add_argument("--count", type=int, default=_env_int("COUNT", 1), help="Worker dequeue batch size.")
    parser.add_argument(
        "--retries", type=int, default=_env_int("RETRIES", 2),
        help="Whole-task retry count for failed labels (orchestrator only).",
    )
    parser.add_argument(
        "--no-repair", action="store_true", default=not _env_bool("REPAIR", True),
        help="Skip repair on validation failure — just re-label instead.",
    )
    parser.add_argument(
        "--redis-url",
        default=os.environ.get("REDIS_URL", "redis://127.0.0.1:6379/0"),
    )
    parser.add_argument("--stream", default=os.environ.get("STREAM", "labeling:tasks"))
    parser.add_argument("--group", default=os.environ.get("GROUP", "labeling-workers"))
    parser.add_argument(
        "--consumer",
        default=os.environ.get("CONSUMER_NAME", f"worker-{os.getpid()}"),
    )
    parser.add_argument(
        "--reclaim-after-ms", type=int,
        default=_env_int("RECLAIM_AFTER_MS", 900_000),
        help="Idle threshold (ms) before a worker reclaims a crashed worker's "
             "task delivery. Must exceed the longest task latency so a slow-but-"
             "alive worker is not stolen from.",
    )
    return parser.parse_args()


def _build_queue(
    *,
    redis_url: str,
    stream: str,
    group: str,
    consumer: str,
    reclaim_after_ms: int,
) -> tuple[Redis, RedisStreamsWorkQueue]:
    """Create the Redis client and the transport queue (no stores — pure transport)."""
    client = Redis.from_url(redis_url)
    queue = RedisStreamsWorkQueue(
        client,
        consumer=consumer,
        stream=stream,
        group=group,
        reclaim_after_ms=reclaim_after_ms,
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
    retries: int,
    repair: bool,
    redis_url: str,
    stream: str,
    group: str,
    consumer: str,
    reclaim_after_ms: int,
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
        reclaim_after_ms=reclaim_after_ms,
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
                retries=retries,
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
        retries=args.retries,
        repair=not args.no_repair,
        redis_url=args.redis_url,
        stream=args.stream,
        group=args.group,
        consumer=args.consumer,
        reclaim_after_ms=args.reclaim_after_ms,
    ))
    if result is not None:
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
