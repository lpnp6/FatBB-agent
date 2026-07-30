"""Tests for the bootstrap corpus sampler."""

from __future__ import annotations

import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from labeling.bootstrap.sample_corpus import (
    assert_not_registered,
    build_manifests,
    cluster_documents,
    persist_labeling_manifest,
)
from labeling.bootstrap.run import prepare_manifest
from labeling.dedup.simhash_store import SimHashDedupStore
from labeling.interfaces.dedup_store import HashStatus


def recipe(name: str) -> str:
    return f"### Ingredients\n- {name}\n\n### Instructions\n1. Cook {name}.\n"


class BootstrapSamplingTests(unittest.TestCase):
    def test_complete_runner_prepares_once_then_reuses_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            corpus = root / "corpus"
            corpus.mkdir()
            for index in range(3):
                (corpus / f"recipe-{index}.md").write_text(recipe(f"r{index}"), encoding="utf-8")
            args = Namespace(
                source_dir=corpus, source_name="corpus", target=2, holdout=1,
                seed=7, threshold=3, output_dir=root / "bootstrap", dedup_db=root / "dedup.db",
            )

            manifest_path, first = prepare_manifest(args)
            same_manifest_path, second = prepare_manifest(args)

            self.assertEqual(first["sampling"], "created")
            self.assertEqual(second, {"sampling": "reused"})
            self.assertEqual(manifest_path, same_manifest_path)
            self.assertEqual(len(manifest_path.read_text(encoding="utf-8").splitlines()), 2)

    def test_sampler_keeps_non_recipe_filenames_from_one_source_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            corpus = root / "corpus"
            corpus.mkdir()
            for index in range(9):
                (corpus / f"recipe-{index}.md").write_text(recipe(f"r{index}"), encoding="utf-8")
            # This would have been removed by the retired filename filter.
            (corpus / "gift-guide.md").write_text("Gift ideas, not a recipe.", encoding="utf-8")

            labeling, holdout, report = build_manifests(
                {"corpus": corpus},
                {"corpus": 1.0},
                target=8,
                holdout=2,
                seed=7,
                threshold=3,
            )

            self.assertEqual(len(labeling), 8)
            self.assertEqual(len(holdout), 2)
            self.assertEqual(report["labeling_by_source"], {"corpus": 8})
            self.assertEqual(report["holdout_by_source"], {"corpus": 2})
            paths = {record["relative_path"] for record in labeling + holdout}
            self.assertIn("gift-guide.md", paths)

    def test_simhash_clusters_duplicate_documents_before_sampling(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = root / "first.md"
            second = root / "second.md"
            first.write_text(recipe("lemon pasta"), encoding="utf-8")
            second.write_text(recipe("lemon pasta"), encoding="utf-8")

            clusters = cluster_documents(
                [("source", first, Path("first.md")), ("source", second, Path("second.md"))],
                threshold=3,
            )

            self.assertEqual(len(clusters), 1)
            self.assertEqual(len(clusters[0].documents), 2)

    def test_labeling_manifest_is_persisted_as_in_flight(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "dedup_store.db"
            records = [{
                "path": "/corpus/recipe.md",
                "recipe_card_hash": "0123456789abcdef",
            }]

            assert_not_registered(records, db_path, threshold=3)
            persist_labeling_manifest(records, db_path, threshold=3)

            store = SimHashDedupStore(db_path)
            try:
                self.assertEqual(store.lookup("0123456789abcdef"), HashStatus.IN_FLIGHT)
            finally:
                store._db.close()
            with self.assertRaisesRegex(ValueError, "already present"):
                assert_not_registered(records, db_path, threshold=3)


if __name__ == "__main__":
    unittest.main()
