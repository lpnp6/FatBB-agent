"""Redis Streams implementation of the labeling work queue."""

from __future__ import annotations

import asyncio
import json
from collections import deque
from typing import Any

from redis.asyncio import Redis
from redis.exceptions import ResponseError

from ..interfaces.checkpoint_store import CheckpointStore
from ..interfaces.dedup_store import DedupEntry, DedupStore
from ..interfaces.work_queue import WorkQueue


class RedisStreamsWorkQueue(WorkQueue):
    """Distribute labeling tasks through one Redis Stream consumer group."""

    def __init__(
        self,
        client: Redis,
        *,
        dedup_store: DedupStore,
        checkpoint: CheckpointStore,
        consumer: str,
        stream: str = "labeling:tasks",
        group: str = "labeling-workers",
        block_ms: int = 1_000,
        reclaim_after_ms: int = 900_000,
        join_poll_seconds: float = 0.1,
    ) -> None:
        self._client = client
        self._dedup_store = dedup_store
        self._checkpoint = checkpoint
        self._consumer = consumer
        self._stream = stream
        self._group = group
        self._block_ms = block_ms
        self._reclaim_after_ms = reclaim_after_ms
        self._join_poll_seconds = join_poll_seconds
        self._outstanding_key = f"{stream}:outstanding"
        self._reclaimed: deque[dict[str, Any]] = deque()

    async def load(self) -> None:
        try:
            await self._client.xgroup_create(
                self._stream, self._group, id="0", mkstream=True,
            )
        except ResponseError as error:
            if "BUSYGROUP" not in str(error):
                raise

    async def enqueue(self, items: list[dict[str, Any]]) -> None:
        if not items:
            return
        async with self._client.pipeline(transaction=True) as pipeline:
            for item in items:
                source_id = str(item["source_id"])
                pipeline.xadd(
                    self._stream,
                    {"payload": json.dumps(item, ensure_ascii=False)},
                )
                pipeline.sadd(self._outstanding_key, source_id)
            await pipeline.execute()

    async def dequeue(self, count: int) -> list[dict[str, Any]]:
        if count <= 0:
            return []

        await self.reclaim_stale()
        tasks = self._take_reclaimed(count)
        if len(tasks) == count:
            return tasks

        messages = await self._client.xreadgroup(
            self._group,
            self._consumer,
            {self._stream: ">"},
            count=count - len(tasks),
            block=self._block_ms,
        )
        for _, entries in messages or []:
            for message_id, fields in entries:
                tasks.append(self._decode(message_id, fields))
        return tasks

    async def join(self) -> None:
        while await self._client.scard(self._outstanding_key):
            await asyncio.sleep(self._join_poll_seconds)

    async def submit_results(
        self, outcomes: list[tuple[dict[str, Any], DedupEntry]],
    ) -> None:
        """Record results, then remove their tasks from the batch barrier."""
        if not outcomes:
            return
        self._dedup_store.register_batch([result for _, result in outcomes])
        await self._checkpoint.mark_completed_batch(
            [str(task["source_id"]) for task, _ in outcomes],
        )
        async with self._client.pipeline(transaction=True) as pipeline:
            for task, _ in outcomes:
                pipeline.srem(self._outstanding_key, str(task["source_id"]))
                pipeline.xack(self._stream, self._group, self._message_id(task))
            await pipeline.execute()

    async def submit_retries(
        self, failures: list[tuple[dict[str, Any], Exception]],
    ) -> None:
        """Requeue failed tasks without releasing the batch barrier."""
        if not failures:
            return
        for task, _ in failures:
            await self._checkpoint.mark_pending(str(task["source_id"]))
            await self._checkpoint.mark_in_flight(
                str(task["source_id"]), str(task["recipe_card_hash"]),
            )
        async with self._client.pipeline(transaction=True) as pipeline:
            for task, error in failures:
                retry_task = self._payload(task)
                retry_task["last_error"] = str(error)
                pipeline.xadd(
                    self._stream,
                    {"payload": json.dumps(retry_task, ensure_ascii=False)},
                )
                pipeline.xack(self._stream, self._group, self._message_id(task))
            await pipeline.execute()

    async def reclaim_stale(self) -> int:
        response = await self._client.xautoclaim(
            self._stream,
            self._group,
            self._consumer,
            self._reclaim_after_ms,
            "0-0",
            count=100,
        )
        _, messages, _ = response
        self._reclaimed.extend(
            self._decode(message_id, fields) for message_id, fields in messages
        )
        return len(messages)

    def _take_reclaimed(self, count: int) -> list[dict[str, Any]]:
        return [self._reclaimed.popleft() for _ in range(min(count, len(self._reclaimed)))]

    @staticmethod
    def _message_id(task: dict[str, Any]) -> str:
        try:
            return str(task["_message_id"])
        except KeyError as error:
            raise ValueError("task was not delivered by this queue") from error

    @staticmethod
    def _payload(task: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in task.items() if key != "_message_id"}

    @classmethod
    def _decode(cls, message_id: str | bytes, fields: dict[str | bytes, str | bytes]) -> dict[str, Any]:
        payload = fields.get("payload") or fields.get(b"payload")
        if payload is None:
            raise ValueError("queue message has no payload")
        if isinstance(payload, bytes):
            payload = payload.decode()
        task = json.loads(payload)
        task["_message_id"] = message_id.decode() if isinstance(message_id, bytes) else message_id
        return task
