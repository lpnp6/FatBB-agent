"""Redis Streams implementation of the labeling work queue."""

from __future__ import annotations

import json
from collections import deque
from typing import Any

from redis.asyncio import Redis
from redis.exceptions import ResponseError

from ..interfaces.work_queue import WorkQueue


class RedisStreamsWorkQueue(WorkQueue):
    """Distribute labeling tasks across two Redis Stream consumer groups.

    Task stream (orchestrator → workers) and result stream (workers →
    orchestrator). The queue is pure transport: it carries payloads but owns no
    dedup/checkpoint state — that lives with the orchestrator.
    """

    def __init__(
        self,
        client: Redis,
        *,
        consumer: str,
        stream: str = "labeling:tasks",
        group: str = "labeling-workers",
        results_stream: str = "labeling:results",
        results_group: str = "labeling-results",
        block_ms: int = 1_000,
        reclaim_after_ms: int = 900_000,
    ) -> None:
        self._client = client
        self._consumer = consumer
        self._stream = stream
        self._group = group
        self._results_stream = results_stream
        self._results_group = results_group
        self._block_ms = block_ms
        self._reclaim_after_ms = reclaim_after_ms
        self._reclaimed: deque[dict[str, Any]] = deque()
        self._reclaimed_results: deque[dict[str, Any]] = deque()

    async def load(self) -> None:
        for stream, group in (
            (self._stream, self._group),
            (self._results_stream, self._results_group),
        ):
            try:
                await self._client.xgroup_create(
                    stream, group, id="0", mkstream=True,
                )
            except ResponseError as error:
                if "BUSYGROUP" not in str(error):
                    raise

    async def enqueue(self, items: list[dict[str, Any]]) -> None:
        if not items:
            return
        async with self._client.pipeline(transaction=True) as pipeline:
            for item in items:
                pipeline.xadd(
                    self._stream,
                    {"payload": json.dumps(item, ensure_ascii=False)},
                )
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

    async def publish_results(
        self, results: list[tuple[dict[str, Any], dict[str, Any]]],
    ) -> None:
        """Publish result payloads and acknowledge their tasks atomically.

        Each element is ``(task, result)``. The task is acknowledged in the same
        transaction as its result is published, so a result is never lost before
        its task is released.
        """
        if not results:
            return
        async with self._client.pipeline(transaction=True) as pipeline:
            for task, result in results:
                pipeline.xadd(
                    self._results_stream,
                    {"payload": json.dumps(result, ensure_ascii=False)},
                )
                pipeline.xack(self._stream, self._group, self._message_id(task))
            await pipeline.execute()

    async def consume_results(self, count: int) -> list[dict[str, Any]]:
        """Return up to *count* result payloads (ack-on-read).

        Results are acknowledged as soon as they are read, so the orchestrator
        never re-reads them; a result lost to a crash between read and persist
        is recovered on the next run because its checkpoint item is still
        IN_FLIGHT.
        """
        if count <= 0:
            return []

        await self.reclaim_stale_results()
        results = self._take_reclaimed_results(count)
        if len(results) == count:
            return results

        messages = await self._client.xreadgroup(
            self._results_group,
            self._consumer,
            {self._results_stream: ">"},
            count=count - len(results),
            block=self._block_ms,
        )
        for _, entries in messages or []:
            ids = [message_id for message_id, _ in entries]
            if ids:
                await self._client.xack(
                    self._results_stream, self._results_group, *ids,
                )
            for message_id, fields in entries:
                results.append(self._decode(message_id, fields))
        return results

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

    async def reclaim_stale_results(self) -> int:
        response = await self._client.xautoclaim(
            self._results_stream,
            self._results_group,
            self._consumer,
            self._reclaim_after_ms,
            "0-0",
            count=100,
        )
        _, messages, _ = response
        ids = [message_id for message_id, _ in messages]
        if ids:
            await self._client.xack(
                self._results_stream, self._results_group, *ids,
            )
        self._reclaimed_results.extend(
            self._decode(message_id, fields) for message_id, fields in messages
        )
        return len(messages)

    def _take_reclaimed(self, count: int) -> list[dict[str, Any]]:
        return [self._reclaimed.popleft() for _ in range(min(count, len(self._reclaimed)))]

    def _take_reclaimed_results(self, count: int) -> list[dict[str, Any]]:
        return [self._reclaimed_results.popleft() for _ in range(min(count, len(self._reclaimed_results)))]

    @staticmethod
    def _message_id(task: dict[str, Any]) -> str:
        try:
            return str(task["_message_id"])
        except KeyError as error:
            raise ValueError("task was not delivered by this queue") from error

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
