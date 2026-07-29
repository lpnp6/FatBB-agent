"""Legacy PostgreSQL repository for configured knowledge bases."""

from __future__ import annotations

from fatbb.domain.knowledge_base import KnowledgeBase, KnowledgeBaseConfig


class PostgresKnowledgeBaseRepository:
    def __init__(self, dsn: str):
        if not dsn:
            raise ValueError("PostgreSQL URL is required.")
        self._dsn = dsn

    def list_knowledge_bases(self) -> list[KnowledgeBase]:
        query = (
            "SELECT id, name, retrieval_type, database_type, database_url, source_type, source_path "
            "FROM knowledge_bases ORDER BY name"
        )
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(query)
            return [self._to_model(row) for row in cursor.fetchall()]

    def create_knowledge_base(self, knowledge_base: KnowledgeBase) -> None:
        query = (
            "INSERT INTO knowledge_bases "
            "(id, name, retrieval_type, database_type, database_url, source_type, source_path) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s)"
        )
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                query,
                (
                    knowledge_base.id, knowledge_base.name,
                    knowledge_base.config.retrieval_type, knowledge_base.config.database_type,
                    knowledge_base.config.database_url, knowledge_base.config.source_type,
                    knowledge_base.source_path,
                ),
            )

    def _connect(self):
        import psycopg
        return psycopg.connect(self._dsn)

    def _to_model(self, row: tuple[object, ...]) -> KnowledgeBase:
        # Rows created before migration 0003 have an empty URL. They are read
        # through a repository built from the URL just entered by the user, so
        # use that URL as a backwards-compatible fallback.
        database_url = str(row[4]) or self._dsn
        return KnowledgeBase(
            id=str(row[0]), name=str(row[1]),
            config=KnowledgeBaseConfig(
                retrieval_type=str(row[2]), database_type=str(row[3]),
                database_url=database_url, source_type=str(row[5])
            ),
            source_path=str(row[6]),
        )
