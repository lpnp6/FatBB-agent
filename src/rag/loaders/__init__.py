"""Document loaders for specific data sources."""

from .sqlite_dedup import SqliteDedupLoader

__all__ = ["SqliteDedupLoader"]
