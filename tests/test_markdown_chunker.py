"""Behavior tests for structure-aware Markdown chunking."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag.chunkers import MarkdownChunker
from rag.models import Document, SourceRef


def document(content: str) -> Document:
    """Build a document fixture with source metadata that chunks must preserve."""
    return Document(
        id="guide",
        content=content,
        source=SourceRef(uri="https://example.com/guide", title="Guide"),
        metadata={"tenant_id": "acme"},
    )


class MarkdownChunkerTests(unittest.TestCase):
    """Verify heading boundaries, size limits, and stable chunk provenance."""

    def test_merges_adjacent_short_sections_and_preserves_heading_path(self) -> None:
        """Small heading sections combine when their combined size is within the cap."""
        chunks = MarkdownChunker(max_chars=200, min_chunk_chars=80).chunk(
            document(
                "# RAG Guide\n\nIntro.\n\n"
                "## Setup\n\nInstall PostgreSQL.\n\n"
                "## Query\n\nUse BM25 to retrieve relevant chunks."
            )
        )

        self.assertEqual(len(chunks), 1)
        self.assertIn("## Setup", chunks[0].content)
        self.assertEqual(chunks[0].metadata["heading_path"], ["RAG Guide"])
        self.assertEqual(chunks[0].source.document_id, "guide")
        self.assertEqual(chunks[0].metadata["tenant_id"], "acme")

    def test_splits_long_sections_without_exceeding_max_chars(self) -> None:
        """Long content is first split at paragraphs and never exceeds the limit."""
        chunks = MarkdownChunker(max_chars=60, min_chunk_chars=20).chunk(
            document(
                "# Long\n\n"
                "First paragraph contains enough words to require splitting.\n\n"
                "Second paragraph also contains enough words to require splitting."
            )
        )

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk.content) <= 60 for chunk in chunks))
        self.assertTrue(all(chunk.metadata["heading_path"] == ["Long"] for chunk in chunks))

    def test_ignores_heading_syntax_inside_fenced_code(self) -> None:
        """A hash-prefixed line inside a fenced code block is not a section boundary."""
        chunks = MarkdownChunker(max_chars=200, min_chunk_chars=1).chunk(
            document("# Guide\n\n```python\n# not a heading\nprint('ok')\n```")
        )

        self.assertEqual(len(chunks), 1)
        self.assertIn("# not a heading", chunks[0].content)

    def test_chunk_ids_are_stable_for_identical_documents(self) -> None:
        """Rebuilding unchanged content produces the same ordered chunk identifiers."""
        chunker = MarkdownChunker(max_chars=100, min_chunk_chars=20)
        content = "# Guide\n\nStable content for repeatable indexing."

        first_ids = [chunk.id for chunk in chunker.chunk(document(content))]
        second_ids = [chunk.id for chunk in chunker.chunk(document(content))]

        self.assertEqual(first_ids, second_ids)


if __name__ == "__main__":
    unittest.main()
