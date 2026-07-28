"""Behavior tests for the SQLite-backed SimHash deduplication store."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from labeling.dedup.simhash_store import SimHashDedupStore
from labeling.interfaces.dedup_store import HashStatus


class SimHashDedupStoreTests(unittest.TestCase):
    """Behavior tests for SimHash fingerprints and their SQLite lifecycle."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "dedup.sqlite3"
        self.store = SimHashDedupStore(self.db_path)

    def tearDown(self) -> None:
        self.store._db.close()
        self.temp_dir.cleanup()

    def test_recipe_card_hash_ignores_non_recipe_content_and_formatting_noise(self) -> None:
        """Only recipe content contributes to a stable, 64-bit fingerprint."""
        base = """\
# Lemon pasta

### Ingredients
- 200g spaghetti
- lemon zest
- parmesan cheese

### Instructions
1. Boil pasta.
2. Toss with lemon and cheese.

### Notes
Published 2024-01-10. Read more at https://example.com/lemon-pasta
"""
        variant = """\
An unrelated introduction with different advertising copy.

### Ingredients
* 200g spaghetti
* lemon zest
* parmesan cheese

### Instructions
1. Boil pasta.
2. Toss with lemon and cheese.

## Reader comments
Updated 2025/12/31: I loved this recipe.
"""

        fingerprint = self.store.recipe_card_hash(base)

        self.assertEqual(fingerprint, self.store.recipe_card_hash(variant))
        self.assertRegex(fingerprint, r"^[0-9a-f]{16}$")

    def test_lookup_returns_status_for_hash_within_threshold(self) -> None:
        """A candidate at the configured Hamming distance is a near duplicate."""
        stored_hash = "0000000000000000"
        self.store.register(stored_hash, "lemon-pasta.md", HashStatus.ACCEPTED)

        # Three changed bits is within the default threshold of three.
        self.assertEqual(self.store.lookup("0000000000000007"), HashStatus.ACCEPTED)

    def test_lookup_rejects_hash_outside_threshold(self) -> None:
        """A shared index block alone is insufficient beyond the Hamming threshold."""
        stored_hash = "0000000000000000"
        self.store.register(stored_hash, "lemon-pasta.md", HashStatus.ACCEPTED)

        # Four changed bits shares index blocks but must fail the final distance check.
        self.assertIsNone(self.store.lookup("000000000000000f"))

    def test_register_creates_one_index_entry_per_block_and_persists_status(self) -> None:
        """Registration writes every block and survives a SQLite reconnection."""
        fingerprint = "0123456789abcdef"
        self.store.register(fingerprint, "pasta.md", HashStatus.IN_FLIGHT)

        index_count = self.store._db.execute(
            "SELECT COUNT(*) FROM simhash_index WHERE hash = ?", (fingerprint,)
        ).fetchone()[0]
        self.assertEqual(index_count, 4)

        self.store.update_status(fingerprint, HashStatus.REJECTED)
        self.store._db.close()
        self.store = SimHashDedupStore(self.db_path)

        self.assertEqual(self.store.lookup(fingerprint), HashStatus.REJECTED)

    def test_clear_in_flight_by_slugs_only_removes_requested_in_flight_records(self) -> None:
        """Crash recovery removes only matching in-flight records and their index rows."""
        retry_hash = "0000000000000000"
        accepted_hash = "ffffffffffffffff"
        other_hash = "aaaaaaaaaaaaaaaa"
        self.store.register(retry_hash, "retry.md", HashStatus.IN_FLIGHT)
        self.store.register(accepted_hash, "retry.md", HashStatus.ACCEPTED)
        self.store.register(other_hash, "other.md", HashStatus.IN_FLIGHT)

        self.store.clear_in_flight_by_slugs({"retry.md"})

        self.assertIsNone(self.store.lookup(retry_hash))
        self.assertEqual(self.store.lookup(accepted_hash), HashStatus.ACCEPTED)
        self.assertEqual(self.store.lookup(other_hash), HashStatus.IN_FLIGHT)

    def test_expire_stale_removes_only_old_in_flight_records(self) -> None:
        """Expiry removes aged in-flight entries without affecting active statuses."""
        stale_hash = "0000000000000000"
        current_hash = "ffffffffffffffff"
        accepted_hash = "aaaaaaaaaaaaaaaa"
        self.store.register(stale_hash, "stale.md", HashStatus.IN_FLIGHT)
        self.store.register(current_hash, "current.md", HashStatus.IN_FLIGHT)
        self.store.register(accepted_hash, "accepted.md", HashStatus.ACCEPTED)
        self.store._db.execute(
            "UPDATE simhashes SET created_at = datetime('now', '-61 minutes') WHERE hash = ?",
            (stale_hash,),
        )
        self.store._db.commit()

        self.assertEqual(self.store.expire_stale(timeout_minutes=60), 1)
        self.assertIsNone(self.store.lookup(stale_hash))
        self.assertEqual(self.store.lookup(current_hash), HashStatus.IN_FLIGHT)
        self.assertEqual(self.store.lookup(accepted_hash), HashStatus.ACCEPTED)


if __name__ == "__main__":
    unittest.main()
