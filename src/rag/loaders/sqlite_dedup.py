"""Load accepted dedup-store rows as graph documents."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing

from ..interfaces.loader import DocumentLoader
from ..models.common import SourceRef
from ..models.document import Document


class SqliteDedupLoader(DocumentLoader):
    """Turn accepted ``simhashes`` rows into :class:`Document` values.

    Column → field mapping:

    - ``source_id`` → ``Document.id`` / ``SourceRef.document_id``
    - ``raw_text``  → ``Document.content``
    - ``output``    → ``Document.metadata["extraction"]`` (JSON-decoded)

    The origin of each document is the dedup store itself; ``source_id`` is the
    row's stable id. ``hash``/``model``/``created_at`` are dropped.
    """

    def __init__(self, path: str):
        self._path = path

    def load(self) -> list[Document]:
        documents: list[Document] = []
        with closing(sqlite3.connect(self._path)) as db:
            rows = db.execute(
                "SELECT source_id, raw_text, output FROM simhashes "
                "WHERE status = 'accepted' AND output IS NOT NULL"
            )
            for source_id, raw_text, output in rows:
                documents.append(
                    Document(
                        id=source_id,
                        content=raw_text,
                        source=SourceRef(document_id=source_id),
                        metadata={"extraction": json.loads(output)},
                    )
                )
        return documents
