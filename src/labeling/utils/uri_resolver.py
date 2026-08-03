"""URIResolver — abstract interface for resolving document URIs to content.

Each supported URI scheme gets a concrete resolver that normalizes the URI
into a stable unique identifier (``source_id``) and produces the raw document
text for downstream processing (dedup, labeling).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from hashlib import blake2b
from pathlib import Path


class URIResolver(ABC):
    """Resolve a document URI into a stable identifier and raw content."""

    @abstractmethod
    def source_id(self, uri: str) -> str:
        """Return a deterministic unique identifier for *uri*.

        The identifier is stable across runs: the same URI always produces the
        same ``source_id``.  Stored in the dedup store to link each hash back
        to its originating document.
        """
        ...

    @abstractmethod
    def resolve(self, uri: str) -> str:
        """Read the raw document text from *uri*."""
        ...

    # ---- discovery ------------------------------------------------------------

    @abstractmethod
    def iter_files(
        self, root: Path | str, *, glob: str = "**/*",
    ) -> Iterator[tuple[str, str]]:
        """Walk *root* and yield ``(uri, source_id)`` for every matching document.

        Each concrete implementation discovers documents in its own namespace
        (local filesystem, S3 bucket, etc.) and yields the canonical URI plus
        the pre-computed ``source_id`` so the caller can register them without
        re-reading the document.

        Args:
            root: Root location to walk (directory path, bucket prefix, …).
            glob: Pattern for filtering (filesystem-style; S3 may ignore).

        Yields:
            Tuples of ``(uri, source_id)``.
        """
        ...


class FileSystemURIResolver(URIResolver):
    """Resolve ``file://`` URIs and local filesystem paths.

    Each file path is hashed with BLAKE2b to produce a stable, collision-
    resistant ``source_id`` independent of absolute path length or encoding.
    """

    def __init__(self, *, base_dir: Path | None = None) -> None:
        self._base_dir = base_dir

    def source_id(self, uri: str) -> str:
        path = self._resolve_path(uri)
        return f"file:{blake2b(str(path).encode(), digest_size=12).hexdigest()}"

    def resolve(self, uri: str) -> str:
        path = self._resolve_path(uri)
        return path.read_text(encoding="utf-8", errors="replace")

    # ---- discovery ------------------------------------------------------------

    def iter_files(
        self, root: Path | str, *, glob: str = "**/*",
    ) -> Iterator[tuple[str, str]]:
        """Walk *root* and yield ``(uri, source_id)`` for every matching file.

        Args:
            root: Directory to walk.
            glob: Pattern passed to :meth:`Path.glob`.  Default ``**/*``
                matches every file recursively.

        Yields:
            Tuples of ``(absolute_path_uri, source_id)``.
        """
        for path in Path(root).glob(glob):
            if path.is_file():
                uri = str(path.resolve())
                yield uri, self.source_id(uri)

    # ---- internal ------------------------------------------------------------

    def _resolve_path(self, uri: str) -> Path:
        """Normalise a URI or path into an absolute :class:`Path`."""
        cleaned = uri.strip()
        if cleaned.startswith("file://"):
            cleaned = cleaned[7:]
        candidate = Path(cleaned)
        if not candidate.is_absolute() and self._base_dir is not None:
            candidate = self._base_dir / candidate
        return candidate.resolve()
