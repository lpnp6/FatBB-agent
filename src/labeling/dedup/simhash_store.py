"""SimHashDedupStore — fuzzy dedup via SimHash + Hamming distance.

Unlike exact-hash stores, this catches near-duplicates: the same recipe
republished with minor formatting changes, different ad blocks, or
slight edits (substitutions, note additions, etc.).

Indexing strategy — Manku block-index (classic web-dedup approach):
    For a 64-bit SimHash and Hamming threshold T=3, partition into T+1=4
    blocks of 16 bits each. By the pigeonhole principle, two hashes within
    distance 3 MUST share at least one identical block.

    We store 4 index rows per entry, keyed on (block_id, block_value).
    lookup() queries candidates from the 4 matching blocks — O(1) per
    query instead of O(n) full-table scan.

    Block width = ceil(64 / (T+1)) = 16 bits per block.
    4 × 65536 possible (block_id, block_value) buckets; with N entries
    (4N index rows), average bucket size = N / 65536.
    At 100k entries: ~1.5 candidates per lookup.

Reference:
    Manku, Jain, Sarma. "Detecting Near-Duplicates for Web Crawling."
    WWW 2007.
"""

from __future__ import annotations

import re
import sqlite3
from hashlib import blake2b
from pathlib import Path

from ..interfaces.dedup_store import DedupStore, HashStatus

# Number of blocks = threshold + 1  (pigeonhole guarantee)
# Block width = 64 // (threshold + 1)
_BLOCKS = 4
_BLOCK_BITS = 16
_BLOCK_MASK = (1 << _BLOCK_BITS) - 1


class SimHashDedupStore(DedupStore):
    """Persistent fuzzy dedup store: SQLite + SimHash + block index.

    Constructor args:
        db_path: Path to the SQLite database file.
        threshold: Maximum Hamming distance (default 3, ~95%+ similarity).
                   Drives block partition count: threshold + 1 blocks.
    """

    def __init__(self, db_path: Path, threshold: int = 3):
        if threshold < 1 or threshold > 15:
            raise ValueError("threshold must be in 1..15")

        self._db = sqlite3.connect(str(db_path))
        self._db.execute("PRAGMA journal_mode=WAL")
        self._threshold = threshold

        # Primary table — one row per registered hash
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS simhashes ("
            "  hash       TEXT PRIMARY KEY,"
            "  source_file TEXT,"
            "  status     TEXT NOT NULL DEFAULT 'in_flight',"
            "  created_at TEXT DEFAULT (datetime('now'))"
            ")"
        )
        # Block index — T+1 rows per entry, O(1) candidate lookup
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS simhash_index ("
            "  block_id    INTEGER NOT NULL,"
            "  block_value INTEGER NOT NULL,"
            "  hash        TEXT NOT NULL REFERENCES simhashes(hash),"
            "  PRIMARY KEY (block_id, block_value, hash)"
            ") WITHOUT ROWID"
        )
        self._db.commit()

    # ---- fingerprint computation ---------------------------------------

    def recipe_card_hash(self, markdown: str) -> str:
        """Build a 64-bit SimHash fingerprint of the recipe card.

        1. Extract ### Ingredients + ### Instructions.
        2. Normalize: strip URLs, dates, markdown formatting.
        3. Tokenize: 2+ char word / CJK sequences.
        4. BLAKE2b token hash (8 bytes) -> 64-bit int.
        5. Bitwise weighted voting -> 64-bit fingerprint (hex).
        """
        text = self._extract_recipe_card(markdown)
        text = self._normalize(text)
        tokens = self._tokenize(text)
        if not tokens:
            return "0" * 16  # degenerate: empty recipe card
        return self._build_fingerprint(tokens)

    # ---- lifecycle -----------------------------------------------------

    def lookup(self, recipe_card_hash: str) -> HashStatus | None:
        """Block-indexed candidate search + Hamming distance check."""
        blocks = self._hash_blocks(recipe_card_hash)

        candidates: set[str] = set()
        for block_id, block_value in enumerate(blocks):
            rows = self._db.execute(
                "SELECT hash FROM simhash_index "
                "WHERE block_id = ? AND block_value = ?",
                (block_id, block_value),
            ).fetchall()
            candidates.update(r[0] for r in rows)

        if not candidates:
            return None

        placeholders = ",".join("?" * len(candidates))
        rows = self._db.execute(
            f"SELECT hash, status FROM simhashes "
            f"WHERE hash IN ({placeholders})",
            tuple(candidates),
        ).fetchall()

        for stored_hash, status in rows:
            if self._hamming(recipe_card_hash, stored_hash) <= self._threshold:
                return HashStatus(status)

        return None

    def register(
        self,
        recipe_card_hash: str,
        source_file: str,
        status: HashStatus,
    ) -> None:
        self._db.execute(
            "INSERT OR REPLACE INTO simhashes (hash, source_file, status) "
            "VALUES (?, ?, ?)",
            (recipe_card_hash, source_file, status.value),
        )
        # Insert T+1 index rows
        for block_id, block_value in enumerate(self._hash_blocks(recipe_card_hash)):
            self._db.execute(
                "INSERT OR IGNORE INTO simhash_index (block_id, block_value, hash) "
                "VALUES (?, ?, ?)",
                (block_id, block_value, recipe_card_hash),
            )
        self._db.commit()

    def update_status(
        self, recipe_card_hash: str, status: HashStatus
    ) -> None:
        self._db.execute(
            "UPDATE simhashes SET status = ? WHERE hash = ?",
            (status.value, recipe_card_hash),
        )
        self._db.commit()

    def expire_stale(self, timeout_minutes: int) -> int:
        # Collect stale hashes
        rows = self._db.execute(
            "SELECT hash FROM simhashes "
            "WHERE status = 'in_flight' "
            "AND datetime(created_at, ? || ' minutes') < datetime('now')",
            (str(timeout_minutes),),
        ).fetchall()
        stale = [r[0] for r in rows]
        if not stale:
            return 0

        placeholders = ",".join("?" * len(stale))
        self._db.execute(
            f"DELETE FROM simhash_index WHERE hash IN ({placeholders})",
            stale,
        )
        self._db.execute(f"DELETE FROM simhashes WHERE hash IN ({placeholders})", stale)
        self._db.commit()
        return len(stale)

    def clear_in_flight_by_slugs(self, slugs: set[str]) -> None:
        if not slugs:
            return
        in_placeholders = ",".join("?" * len(slugs))
        rows = self._db.execute(
            f"SELECT hash FROM simhashes "
            f"WHERE status = 'in_flight' AND source_file IN ({in_placeholders})",
            tuple(slugs),
        ).fetchall()
        stale = [r[0] for r in rows]
        if not stale:
            return

        out_placeholders = ",".join("?" * len(stale))
        self._db.execute(
            f"DELETE FROM simhash_index WHERE hash IN ({out_placeholders})",
            stale,
        )
        self._db.execute(
            f"DELETE FROM simhashes WHERE hash IN ({out_placeholders})",
            stale,
        )
        self._db.commit()

    # ---- SimHash internals ---------------------------------------------

    @staticmethod
    def _extract_recipe_card(markdown: str) -> str:
        """Extract recipe card, handling both blog and print formats.

        Full-page blogs use ### Ingredients / ### Instructions headers.
        wprm_print has no headers — the entire file IS the recipe card.

        For full-page, we stop after Instructions to exclude recipe notes,
        nutrition tables, and user comments — these vary between
        comment-page variants of the same recipe.
        """
        ingredients = ""
        instructions = ""

        m = re.search(
            r"###\s*Ingredients\b.*?(?=###\s*Instructions\b)",
            markdown, re.DOTALL | re.IGNORECASE,
        )
        if m:
            ingredients = m.group(0)
            m2 = re.search(
                r"###\s*Instructions\b.*",
                markdown, re.DOTALL | re.IGNORECASE,
            )
            if m2:
                block = m2.group(0)
                # Cut at the next section boundary to exclude notes/comments
                end = re.search(
                    r"\n(?=###\s|\n##\s)", block, re.DOTALL,
                )
                instructions = block[: end.start()] if end else block

        if ingredients or instructions:
            return ingredients + "\n" + instructions

        # No Markdown sections — the whole file is the recipe card (wprm_print)
        return markdown

    @staticmethod
    def _normalize(text: str) -> str:
        text = text.lower()
        text = re.sub(r"https?://\S+", " ", text)
        text = re.sub(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}", " ", text)
        text = re.sub(r"[`*_>#\[\]()]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"[\w぀-ヿ一-鿿]{2,}", text)

    def _build_fingerprint(self, tokens: list[str]) -> str:
        weights = [0] * 64
        for token in tokens:
            h = int.from_bytes(
                blake2b(token.encode("utf-8"), digest_size=8).digest(),
                "big",
            )
            for bit in range(64):
                if h & (1 << bit):
                    weights[bit] += 1
                else:
                    weights[bit] -= 1
        fingerprint = 0
        for bit in range(64):
            if weights[bit] >= 0:
                fingerprint |= 1 << bit
        return f"{fingerprint:016x}"

    @staticmethod
    def _hamming(hash_a: str, hash_b: str) -> int:
        a, b = int(hash_a, 16), int(hash_b, 16)
        return (a ^ b).bit_count()

    @staticmethod
    def _hash_blocks(hash_hex: str) -> tuple[int, ...]:
        """Split 64-bit hash into T+1 blocks of 16 bits."""
        value = int(hash_hex, 16)
        return tuple(
            (value >> (i * _BLOCK_BITS)) & _BLOCK_MASK
            for i in range(_BLOCKS)
        )
