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
    """Resolve a document identifier into a stable source_id and raw content."""

    @abstractmethod
    def source_id(self, uri: str) -> str:
        """Return a deterministic unique identifier for *uri*.

        The identifier is stable across runs: the same URI always produces the
        same ``source_id``.  Stored in the dedup store to link each hash back
        to its originating document.
        """
        ...

    @abstractmethod
    def resolve(self, source_id: str) -> str:
        """Read the raw document text identified by *source_id*.

        *source_id* may be a raw URI/path or a ``source_id`` produced by
        :meth:`source_id` or :meth:`iter_files`.  Concrete implementations
        decide how to map the identifier back to content.
        """
        ...

    # ---- discovery ------------------------------------------------------------

    @abstractmethod
    def iter_files(
        self, root: Path | str, *, glob: str = "**/*",
    ) -> Iterator[str]:
        """Walk *root* and yield a ``source_id`` for every matching document.

        Each concrete implementation discovers documents in its own namespace
        (local filesystem, S3 bucket, etc.).  The yielded ``source_id`` can
        be passed directly to :meth:`resolve` to read the document content.

        Args:
            root: Root location to walk (directory path, bucket prefix, …).
            glob: Pattern for filtering (filesystem-style; S3 may ignore).

        Yields:
            ``source_id`` strings.
        """
        ...


class FileSystemURIResolver(URIResolver):
    """Resolve local filesystem paths.

    Each file path is hashed with BLAKE2b to produce a stable, collision-
    resistant ``source_id``.  During :meth:`iter_files` the resolver builds
    an internal mapping so ``source_id`` → path resolution is O(1).
    """

    def __init__(self, *, base_dir: Path | None = None) -> None:
        self._base_dir = base_dir
        self._path_of: dict[str, Path] = {}

    def source_id(self, uri: str) -> str:
        path = self._resolve_path(uri)
        return f"file:{blake2b(str(path).encode(), digest_size=12).hexdigest()}"

    def resolve(self, source_id: str) -> str:
        # Fast path: lookup in the iter_files mapping.
        path = self._path_of.get(source_id)
        if path is not None:
            return path.read_text(encoding="utf-8", errors="replace")
        # Fallback: treat as a raw path / file:// URI.
        return self._resolve_path(source_id).read_text(encoding="utf-8", errors="replace")

    # ---- discovery ------------------------------------------------------------

    def iter_files(
        self, root: Path | str, *, glob: str = "**/*",
    ) -> Iterator[str]:
        """Walk *root* and yield a ``source_id`` for every matching file.

        The internal mapping is populated during iteration so subsequent
        :meth:`resolve` calls are O(1).
        """
        for path in Path(root).glob(glob):
            if path.is_file():
                resolved = path.resolve()
                sid = self.source_id(str(resolved))
                self._path_of[sid] = resolved
                yield sid

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
