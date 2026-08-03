"""URIResolver — abstract interface for resolving document URIs to content.

Each supported URI scheme gets a concrete resolver that normalizes the URI
into a stable unique identifier (``source_id``) and produces the raw document
text for downstream processing (dedup, labeling).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
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
