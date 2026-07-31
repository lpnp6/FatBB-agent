"""Storage adapter implementations."""

from .postgres.postgres_bm25_search_store import PostgresBM25SearchStore
from .postgres.postgres_vector_search_store import PostgresVectorSearchStore

PostgresTextChunkStore = PostgresBM25SearchStore

__all__ = [
    "PostgresBM25SearchStore",
    "PostgresTextChunkStore",
    "PostgresVectorSearchStore",
]
