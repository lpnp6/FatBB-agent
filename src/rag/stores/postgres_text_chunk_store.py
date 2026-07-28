"""PostgreSQL implementation of the text-chunk storage port."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict
import json

from ..interfaces.stores import BM25SearchStore, ScoredTextChunk
from ..models.common import SourceRef
from ..models.document import TextChunk


class PostgresTextChunkStore(BM25SearchStore):
    """Persist chunks and execute pg_search BM25 queries through psycopg 3.

    ``psycopg`` is imported only when a database operation occurs, so model and
    retriever unit tests remain dependency-free. Its schema is created by the
    deployment migration in
    ``migrations/postgres/0001_create_rag_text_chunks.sql``.
    """

    def __init__(self, dsn: str):
        if not dsn:
            raise ValueError("dsn cannot be empty")
        self._dsn = dsn

    def list_chunks(self, *, filters: Mapping[str, object]) -> list[TextChunk]:
        query = (
            "SELECT id, document_id, content, chunk_index, source, start_offset, "
            "end_offset, metadata FROM rag_text_chunks"
        )
        parameters: tuple[object, ...] = ()
        if filters:
            query += " WHERE metadata @> %s::jsonb"
            parameters = (json.dumps(dict(filters)),)
        query += " ORDER BY id"

        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(query, parameters)
            return [self._to_chunk(row) for row in cursor.fetchall()]

    def search_bm25(
        self,
        query_text: str,
        *,
        top_k: int,
        filters: Mapping[str, object],
    ) -> list[ScoredTextChunk]:
        """Use pg_search to score, sort, and limit matching text chunks."""
        query = (
            "SELECT id, document_id, content, chunk_index, source, start_offset, "
            "end_offset, metadata, paradedb.score(id) AS score "
            "FROM rag_text_chunks WHERE content @@@ %s"
        )
        parameters: list[object] = [query_text]
        if filters:
            query += " AND metadata @> %s::jsonb"
            parameters.append(json.dumps(dict(filters)))
        query += " ORDER BY score DESC, id ASC LIMIT %s"
        parameters.append(top_k)

        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(query, parameters)
            return [
                ScoredTextChunk(
                    chunk=self._to_chunk(row[:8]),
                    score=self._required_float(row[8], field="score"),
                )
                for row in cursor.fetchall()
            ]

    def upsert_chunks(self, chunks: Sequence[TextChunk]) -> None:
        if not chunks:
            return
        query = """
            INSERT INTO rag_text_chunks (
                id, document_id, content, chunk_index, source, start_offset,
                end_offset, metadata
            ) VALUES (%s, %s, %s, %s, %s::jsonb, %s, %s, %s::jsonb)
            ON CONFLICT (id) DO UPDATE SET
                document_id = EXCLUDED.document_id,
                content = EXCLUDED.content,
                chunk_index = EXCLUDED.chunk_index,
                source = EXCLUDED.source,
                start_offset = EXCLUDED.start_offset,
                end_offset = EXCLUDED.end_offset,
                metadata = EXCLUDED.metadata
        """
        values = [
            (
                chunk.id,
                chunk.document_id,
                chunk.content,
                chunk.index,
                json.dumps(asdict(chunk.source)),
                chunk.start_offset,
                chunk.end_offset,
                json.dumps(chunk.metadata),
            )
            for chunk in chunks
        ]
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.executemany(query, values)

    def delete_chunks(self, chunk_ids: Sequence[str]) -> None:
        if not chunk_ids:
            return
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute("DELETE FROM rag_text_chunks WHERE id = ANY(%s)", (list(chunk_ids),))

    def _connect(self):
        try:
            import psycopg
        except ImportError as error:
            raise RuntimeError(
                "PostgreSQL support requires psycopg. Install dependencies with "
                "`pip install -r requirements.txt`."
            ) from error
        return psycopg.connect(self._dsn)

    @staticmethod
    def _to_chunk(row: tuple[object, ...]) -> TextChunk:
        source_data = PostgresTextChunkStore._json_object(row[4], field="source")
        metadata = PostgresTextChunkStore._json_object(row[7], field="metadata")
        return TextChunk(
            id=str(row[0]),
            document_id=str(row[1]),
            content=str(row[2]),
            index=PostgresTextChunkStore._required_int(row[3], field="chunk_index"),
            source=SourceRef(
                document_id=PostgresTextChunkStore._optional_string(
                    source_data.get("document_id"), field="source.document_id"
                ),
                uri=PostgresTextChunkStore._optional_string(
                    source_data.get("uri"), field="source.uri"
                ),
                title=PostgresTextChunkStore._optional_string(
                    source_data.get("title"), field="source.title"
                ),
                locator=PostgresTextChunkStore._optional_string(
                    source_data.get("locator"), field="source.locator"
                ),
            ),
            start_offset=(
                PostgresTextChunkStore._required_int(row[5], field="start_offset")
                if row[5] is not None
                else None
            ),
            end_offset=(
                PostgresTextChunkStore._required_int(row[6], field="end_offset")
                if row[6] is not None
                else None
            ),
            metadata=metadata,
        )

    @staticmethod
    def _required_int(value: object, *, field: str) -> int:
        """Narrow an untyped PostgreSQL column value to an integer."""
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        raise TypeError(f"PostgreSQL column {field!r} must be an integer")

    @staticmethod
    def _required_float(value: object, *, field: str) -> float:
        """Narrow a numeric PostgreSQL column value to a Python float."""
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        try:
            return float(str(value))
        except ValueError as error:
            raise TypeError(
                f"PostgreSQL column {field!r} must be numeric"
            ) from error

    @staticmethod
    def _optional_string(value: object, *, field: str) -> str | None:
        """Narrow an optional JSON object field to a string."""
        if value is None or isinstance(value, str):
            return value
        raise TypeError(f"PostgreSQL JSON field {field!r} must be a string or null")

    @staticmethod
    def _json_object(value: object, *, field: str) -> dict[str, object]:
        """Decode a JSON/JSONB database value and require an object payload."""
        decoded = json.loads(value) if isinstance(value, str) else value
        if not isinstance(decoded, Mapping):
            raise TypeError(f"PostgreSQL column {field!r} must contain a JSON object")
        return {str(key): item for key, item in decoded.items()}
