"""Tests for PostgreSQL migration execution."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag.stores import PostgresBM25SearchStore


class RecordingCursor:
    def __init__(self) -> None:
        self.statements: list[str] = []

    def __enter__(self) -> RecordingCursor:
        return self

    def __exit__(self, *args: object) -> None:
        pass

    def execute(self, statement: str) -> None:
        self.statements.append(statement)


class RecordingConnection:
    def __init__(self, cursor: RecordingCursor) -> None:
        self._cursor = cursor

    def __enter__(self) -> RecordingConnection:
        return self

    def __exit__(self, *args: object) -> None:
        pass

    def cursor(self) -> RecordingCursor:
        return self._cursor


class PostgresMigrationTests(unittest.TestCase):
    def test_migrate_applies_every_script_with_the_configured_table_name(self) -> None:
        cursor = RecordingCursor()
        store = PostgresBM25SearchStore("postgresql://example", table_name="tenant_chunks")
        store._connect = lambda: RecordingConnection(cursor)  # type: ignore[method-assign]

        store.migrate()

        self.assertEqual(len(cursor.statements), 3)
        self.assertTrue(all("{{table_name}}" not in statement for statement in cursor.statements))
        self.assertTrue(all('"tenant_chunks"' in statement for statement in cursor.statements))
        self.assertIn('"tenant_chunks_embedding_hnsw_idx"', cursor.statements[-1])


if __name__ == "__main__":
    unittest.main()
