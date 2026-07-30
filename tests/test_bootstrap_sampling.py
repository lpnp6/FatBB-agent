"""Tests for the bootstrap corpus sampler."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from labeling.bootstrap.sample_corpus import (
    assert_not_registered,
    build_manifests,
    cluster_documents,
    persist_labeling_manifest,
)
from labeling.dedup.simhash_store import SimHashDedupStore
from labeling.interfaces.dedup_store import HashStatus


def recipe(name: str) -> str:
    return f"### Ingredients\n- {name}\n\n### Instructions\n1. Cook {name}.\n"


class BootstrapSamplingTests(unittest.TestCase):
    def test_sampler_keeps_non_recipe_filenames_and_preserves_source_quota(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            recipetineats = root / "recipetineats"
            wellplated = root / "wellplated"
            recipetineats.mkdir()
            wellplated.mkdir()
            for index in range(3):
                (recipetineats / f"recipe-{index}.md").write_text(recipe(f"r{index}"), encoding="utf-8")
            # This would have been removed by the retired filename filter.
            (recipetineats / "gift-guide.md").write_text("Gift ideas, not a recipe.", encoding="utf-8")
            for index in range(6):
                (wellplated / f"recipe-{index}.md").write_text(recipe(f"w{index}"), encoding="utf-8")

            labeling, holdout, report = build_manifests(
                {"recipetineats": recipetineats, "wellplated": wellplated},
                {"recipetineats": 0.4, "wellplated": 0.6},
                target=8,
                holdout=2,
                seed=7,
                threshold=3,
            )

            self.assertEqual(len(labeling), 8)
            self.assertEqual(len(holdout), 2)
            self.assertEqual(report["labeling_by_source"], {"recipetineats": 3, "wellplated": 5})
            self.assertEqual(report["holdout_by_source"], {"recipetineats": 1, "wellplated": 1})
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
