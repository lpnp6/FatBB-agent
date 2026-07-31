"""PostgreSQL implementation of the text-chunk storage port."""

from __future__ import annotations

from collections.abc import Mapping, Sequence, Callable
from dataclasses import asdict
import json
import logging
from pathlib import Path
import re
from typing import LiteralString

from ...interfaces.stores import BM25SearchStore
from ...models.common import SourceRef
from ...models.document import ScoredTextChunk, TextChunk


logger = logging.getLogger(__name__)


class PostgresBM25SearchStore(BM25SearchStore):
    """Persist chunks and execute pg_search BM25 queries through psycopg 3.

    ``psycopg`` is imported only when a database operation occurs, so model and
    retriever unit tests remain dependency-free. Its schema is created by the
    deployment migration in
    ``migrations/postgres/0001_create_rag_text_chunks.sql``.
    """

    def __init__(self, dsn: str, *, table_name: str = "rag_text_chunks"):
        if not dsn:
            raise ValueError("dsn cannot be empty")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table_name):
            raise ValueError("table_name must be a valid unqualified SQL identifier")
        self._dsn = dsn
        self._table_name = table_name

    def check_connection(self) -> None:
        """Open and validate a PostgreSQL connection without changing data."""
        logger.info("Checking PostgreSQL database connection")
        try:
            with self._connect() as connection, connection.cursor() as cursor:
                cursor.execute("SELECT 1")
        except Exception:
            logger.exception("PostgreSQL database connection check failed")
            raise
        logger.info("PostgreSQL database connection verified")

    def migrate(self) -> None:
        """Apply the bundled PostgreSQL migrations to this store's table."""
        with self._connect() as connection, connection.cursor() as cursor:
            for path in self._migration_paths():
                cursor.execute(self._migration_sql(path.read_text(encoding="utf-8")))

    def _migration_paths(self) -> list[Path]:
        migrations = Path(__file__).resolve().parents[4] / "migrations" / "postgres"
        return sorted(migrations.glob("*.sql"))

    def _migration_sql(self, script: str) -> str:
        """Substitute the validated table name into a migration template."""
        return script.replace("{{table_name}}", self._table_name)

    def list_chunks(self, *, filters: Mapping[str, object]) -> list[TextChunk]:
        query = self._query(
            "SELECT id, document_id, content, chunk_index, source, start_offset, "
            "end_offset, metadata FROM {table}"
        )
        parameters: tuple[object, ...] = ()
        if filters:
            query += self._query(" WHERE metadata @> %s::jsonb")
            parameters = (json.dumps(dict(filters)),)
        query += self._query(" ORDER BY id")

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
        query = self._query(
            "SELECT id, document_id, content, chunk_index, source, start_offset, "
            "end_offset, metadata, paradedb.score(id) AS score "
            "FROM {table} WHERE content @@@ %s"
        )
        parameters: list[object] = [query_text]
        if filters:
            query += self._query(" AND metadata @> %s::jsonb")
            parameters.append(json.dumps(dict(filters)))
        query += self._query(" ORDER BY score DESC, id ASC LIMIT %s")
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
        with self._connect() as connection, connection.cursor() as cursor:
            self._upsert_chunks(cursor, chunks)

    def delete_chunks(self, chunk_ids: Sequence[str]) -> None:
        if not chunk_ids:
            return
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                self._query("DELETE FROM {table} WHERE id = ANY(%s)"),
                (list(chunk_ids),),
            )

    def replace_document_chunks(
        self, entries: Sequence[tuple[str, Sequence[TextChunk]]], *,
        on_progress: Callable[[str, int, int], None] | None = None,
    ) -> None:
        """Replace chunks for multiple documents in a single connection.

        Existing IDs are preserved until replacement rows have been prepared;
        rows no longer emitted by the chunker are removed before commit.
        """
        if not entries:
            return

        # Validate before opening a connection.
        for document_id, chunks in entries:
            if any(chunk.document_id != document_id for chunk in chunks):
                raise ValueError("every replacement chunk must belong to document_id")

        total = len(entries)
        with self._connect() as connection, connection.cursor() as cursor:
            for idx, (document_id, chunks) in enumerate(entries):
                if on_progress is not None:
                    on_progress("Writing to database", idx + 1, total)
                if chunks:
                    cursor.execute(
                        self._query("DELETE FROM {table} ")
                        + self._query(
                            "WHERE document_id = %s AND NOT (id = ANY(%s))",
                        ),
                        (document_id, [chunk.id for chunk in chunks]),
                    )
                    self._upsert_chunks(cursor, chunks)
                else:
                    cursor.execute(
                        self._query("DELETE FROM {table} WHERE document_id = %s"),
                        (document_id,),
                    )

    def delete_by_document_ids(self, document_ids: Sequence[str]) -> None:
        """Remove all chunk rows associated with deleted source documents."""
        if not document_ids:
            return
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                self._query("DELETE FROM {table} WHERE document_id = ANY(%s)"),
                (list(document_ids),),
            )

    def _upsert_chunks(self, cursor: object, chunks: Sequence[TextChunk]) -> None:
        """Execute the shared chunk upsert statement using an open cursor."""
        query = self._query(
            """
                INSERT INTO {table} (
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
        )
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
        # psycopg's cursor type is intentionally kept out of this module's
        # import surface so pure domain tests do not require the driver.
        cursor.executemany(query, values)  # type: ignore[attr-defined]

    def _connect(self):
        try:
            import psycopg
        except ImportError as error:
            raise RuntimeError(
                "PostgreSQL support requires psycopg. Install dependencies with "
                "`pip install -r requirements.txt`."
            ) from error
        return psycopg.connect(self._dsn)

    def _query(self, template: LiteralString):
        """Safely interpolate the configured table identifier into SQL."""
        try:
            from psycopg import sql
        except ImportError as error:
            raise RuntimeError(
                "PostgreSQL support requires psycopg. Install dependencies with "
                "`pip install -r requirements.txt`."
            ) from error
        return sql.SQL(template).format(table=sql.Identifier(self._table_name))

    @staticmethod
    def _to_chunk(row: tuple[object, ...]) -> TextChunk:
        source_data = PostgresBM25SearchStore._json_object(row[4], field="source")
        metadata = PostgresBM25SearchStore._json_object(row[7], field="metadata")
        return TextChunk(
            id=str(row[0]),
            document_id=str(row[1]),
            content=str(row[2]),
            index=PostgresBM25SearchStore._required_int(row[3], field="chunk_index"),
            source=SourceRef(
                document_id=PostgresBM25SearchStore._optional_string(
                    source_data.get("document_id"), field="source.document_id"
                ),
                uri=PostgresBM25SearchStore._optional_string(
                    source_data.get("uri"), field="source.uri"
                ),
                title=PostgresBM25SearchStore._optional_string(
                    source_data.get("title"), field="source.title"
                ),
                locator=PostgresBM25SearchStore._optional_string(
                    source_data.get("locator"), field="source.locator"
                ),
            ),
            start_offset=(
                PostgresBM25SearchStore._required_int(row[5], field="start_offset")
                if row[5] is not None
                else None
            ),
            end_offset=(
                PostgresBM25SearchStore._required_int(row[6], field="end_offset")
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
