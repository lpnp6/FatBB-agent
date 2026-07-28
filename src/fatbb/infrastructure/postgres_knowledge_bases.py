"""PostgreSQL repository for configured knowledge bases."""

from __future__ import annotations

from fatbb.domain.knowledge_base import KnowledgeBase, KnowledgeBaseConfig


class PostgresKnowledgeBaseRepository:
    def __init__(self, dsn: str):
        if not dsn:
            raise ValueError("DATABASE_URL is required.")
        self._dsn = dsn

    def list(self) -> list[KnowledgeBase]:
        query = (
            "SELECT id, name, retrieval_type, database_type, source_type, source_path "
            "FROM knowledge_bases ORDER BY name"
        )
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(query)
            return [self._to_model(row) for row in cursor.fetchall()]

    def create(self, knowledge_base: KnowledgeBase) -> None:
        query = (
            "INSERT INTO knowledge_bases "
            "(id, name, retrieval_type, database_type, source_type, source_path) "
            "VALUES (%s, %s, %s, %s, %s, %s)"
        )
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                query,
                (
                    knowledge_base.id, knowledge_base.name,
                    knowledge_base.config.retrieval_type, knowledge_base.config.database_type,
                    knowledge_base.config.source_type, knowledge_base.source_path,
                ),
            )

    def _connect(self):
        import psycopg
        return psycopg.connect(self._dsn)

    @staticmethod
    def _to_model(row: tuple[object, ...]) -> KnowledgeBase:
        return KnowledgeBase(
            id=str(row[0]), name=str(row[1]),
            config=KnowledgeBaseConfig(
                retrieval_type=str(row[2]), database_type=str(row[3]), source_type=str(row[4])
            ),
            source_path=str(row[5]),
        )
