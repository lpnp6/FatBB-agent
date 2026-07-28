"""Contract tests for the Chunker abstraction."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag.interfaces import Chunker


class ChunkerInterfaceTests(unittest.TestCase):
    """Ensure Chunker remains an abstract port until a strategy implements it."""

    def test_chunker_requires_a_chunk_implementation(self) -> None:
        """The shared interface cannot be instantiated without `chunk()`."""
        with self.assertRaises(TypeError):
            Chunker()


if __name__ == "__main__":
    unittest.main()
