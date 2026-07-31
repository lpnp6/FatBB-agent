"""Tests for capability construction from the versioned TOML catalog."""

from __future__ import annotations

import unittest
from pathlib import Path
import tempfile
from fatbb.application.registry import CapabilityRegistry


class RegistryPluginTests(unittest.TestCase):
    def test_catalog_builds_adapters_from_stable_type_keys(self) -> None:
        registry = CapabilityRegistry(_catalog_path())

        self.assertEqual(registry.knowledge_base("bm25", "pg").type, "bm25")
        self.assertEqual(registry.knowledge_base("vector", "pg").type, "vector")
        self.assertEqual(registry.importer("file_path").type, "file_path")

    def test_unknown_type_is_rejected(self) -> None:
        registry = CapabilityRegistry(_catalog_path())

        with self.assertRaisesRegex(ValueError, "Unsupported retrieval type"):
            registry.knowledge_base("unknown", "pg")



def _catalog_path() -> Path:
    return Path(__file__).parents[1] / "src" / "fatbb" / "config" / "kb.toml"
