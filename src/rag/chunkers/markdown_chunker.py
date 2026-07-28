"""Structure-aware chunking for Markdown knowledge documents."""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
import re

from ..interfaces.chunker import Chunker
from ..models.document import Document, TextChunk

_HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)(?:\s+#+)?\s*$")
_FENCE_PATTERN = re.compile(r"^\s*(`{3,}|~{3,})")


@dataclass(frozen=True)
class _Section:
    """A Markdown heading block before size-based merging or splitting."""

    content: str
    heading_path: tuple[str, ...]


class MarkdownChunker(Chunker):
    """Chunk Markdown by headings, merge short sections, and cap chunk size.

    Headings remain in chunk content so lexical retrieval can match them. The
    corresponding hierarchy is stored in each chunk's ``heading_path`` metadata.
    This implementation deliberately does not filter boilerplate or add overlap.
    """

    def __init__(self, *, max_chars: int = 2_500, min_chunk_chars: int = 300):
        if max_chars < 1:
            raise ValueError("max_chars must be at least 1")
        if min_chunk_chars < 1:
            raise ValueError("min_chunk_chars must be at least 1")
        if min_chunk_chars > max_chars:
            raise ValueError("min_chunk_chars cannot exceed max_chars")
        self._max_chars = max_chars
        self._min_chunk_chars = min_chunk_chars

    def chunk(self, document: Document) -> list[TextChunk]:
        """Return stable chunks in source order for one Markdown document."""
        sections = self._sections(document.content)
        pieces = [
            piece
            for section in sections
            for piece in self._split_to_max_size(section)
        ]
        merged = self._merge_short_pieces(pieces)
        source = replace(document.source, document_id=document.id)

        return [
            TextChunk(
                id=self._chunk_id(document.id, index, section.content),
                document_id=document.id,
                content=section.content,
                index=index,
                source=source,
                metadata={
                    **document.metadata,
                    "chunker": "markdown",
                    "chunker_version": "v1",
                    "heading_path": list(section.heading_path),
                },
            )
            for index, section in enumerate(merged)
        ]

    def _sections(self, markdown: str) -> list[_Section]:
        """Split source at ATX headings while ignoring heading-like fenced code."""
        sections: list[_Section] = []
        lines: list[str] = []
        heading_path: list[str] = []
        fence_marker: str | None = None

        def flush() -> None:
            content = "".join(lines).strip()
            if content:
                sections.append(_Section(content=content, heading_path=tuple(heading_path)))

        for line in markdown.splitlines(keepends=True):
            fence_match = _FENCE_PATTERN.match(line)
            if fence_match:
                marker = fence_match.group(1)
                if fence_marker is None:
                    fence_marker = marker[0]
                elif marker[0] == fence_marker:
                    fence_marker = None
                lines.append(line)
                continue

            heading_match = None if fence_marker else _HEADING_PATTERN.match(line)
            if heading_match is None:
                lines.append(line)
                continue

            flush()
            lines = [line]
            level = len(heading_match.group(1))
            title = heading_match.group(2).strip()
            heading_path = heading_path[: level - 1]
            heading_path.append(title)

        flush()
        return sections

    def _split_to_max_size(self, section: _Section) -> list[_Section]:
        """Prefer paragraph boundaries, then whitespace, before hard splitting."""
        if len(section.content) <= self._max_chars:
            return [section]

        paragraphs = re.split(r"\n\s*\n", section.content)
        pieces: list[str] = []
        current = ""
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
            if len(candidate) <= self._max_chars:
                current = candidate
                continue
            if current:
                pieces.append(current)
                current = ""
            pieces.extend(self._split_hard(paragraph))

        if current:
            pieces.append(current)
        return [
            _Section(content=piece, heading_path=section.heading_path)
            for piece in pieces
        ]

    def _split_hard(self, text: str) -> list[str]:
        """Split an oversized paragraph at whitespace when possible."""
        pieces: list[str] = []
        remaining = text
        while len(remaining) > self._max_chars:
            boundary = remaining.rfind(" ", 0, self._max_chars + 1)
            if boundary <= 0:
                boundary = self._max_chars
            pieces.append(remaining[:boundary].strip())
            remaining = remaining[boundary:].strip()
        if remaining:
            pieces.append(remaining)
        return pieces

    def _merge_short_pieces(self, pieces: list[_Section]) -> list[_Section]:
        """Merge a short piece into its following neighbor when it fits."""
        merged: list[_Section] = []
        pending: _Section | None = None

        for piece in pieces:
            if pending is None:
                pending = piece
                continue
            combined = f"{pending.content}\n\n{piece.content}"
            if len(pending.content) < self._min_chunk_chars and len(combined) <= self._max_chars:
                pending = _Section(
                    content=combined,
                    heading_path=pending.heading_path,
                )
                continue
            merged.append(pending)
            pending = piece

        if pending is not None:
            merged.append(pending)
        return merged

    @staticmethod
    def _chunk_id(document_id: str, index: int, content: str) -> str:
        """Produce a stable ID for one document position and its exact content."""
        digest = sha256(content.encode("utf-8")).hexdigest()[:16]
        return f"{document_id}:{index}:{digest}"
