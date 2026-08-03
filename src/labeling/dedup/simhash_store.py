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

from ..interfaces.checkpoint_store import CheckpointStore
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

    def __init__(self, db_path: Path, threshold: int = 3, *, checkpoint: CheckpointStore | None = None):
        if threshold < 1 or threshold > 15:
            raise ValueError("threshold must be in 1..15")

        self._db = sqlite3.connect(str(db_path))
        self._db.execute("PRAGMA journal_mode=WAL")
        self._threshold = threshold
        self._checkpoint = checkpoint

        # Primary table — one row per registered hash
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS simhashes ("
            "  hash       TEXT PRIMARY KEY,"
            "  status     TEXT NOT NULL DEFAULT 'in_flight',"
            "  raw_text   TEXT,"
            "  model      TEXT,"
            "  output     TEXT,"
            "  created_at TEXT DEFAULT (datetime('now'))"
            ")"
        )
        # Migrate existing tables that lack newer columns
        for col in ("raw_text", "model", "output"):
            try:
                self._db.execute(f"ALTER TABLE simhashes ADD COLUMN {col} TEXT")
            except sqlite3.OperationalError:
                pass  # column already exists
        # Drop legacy source_file column (no longer needed — raw_text supersedes it)
        try:
            self._db.execute("ALTER TABLE simhashes DROP COLUMN source_file")
        except sqlite3.OperationalError:
            pass  # column already dropped or doesn't exist
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

    # ---- embedded store ---------------------------------------------------

    @property
    def checkpoint(self) -> CheckpointStore:
        """The CheckpointStore embedded in this dedup store."""
        if self._checkpoint is None:
            raise RuntimeError("SimHashDedupStore was not constructed with a CheckpointStore")
        return self._checkpoint

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
        status: HashStatus,
        *,
        raw_text: str | None = None,
        model: str | None = None,
        output: str | None = None,
    ) -> None:
        """Persist a hash with its initial status.

        Writes to two tables in one transaction:
            1. simhashes: one row — the authoritative record.
            2. simhash_index: T+1 rows — one per 16-bit block of the 64-bit
               SimHash. This is the Manku block index that enables O(1)
               candidate lookup instead of full-table scan.

        INSERT OR REPLACE on simhashes ensures re-registering the same hash
        (e.g. after expire_stale cleanup) overwrites the old status.
        INSERT OR IGNORE on simhash_index avoids duplicate index rows —
        the same (block_id, block_value, hash) tuple may already exist
        if the hash was previously registered and expired/recovered.
        """
        self._db.execute(
            "INSERT OR REPLACE INTO simhashes (hash, status, raw_text, model, output) "
            "VALUES (?, ?, ?, ?, ?)",
            (recipe_card_hash, status.value, raw_text, model, output),
        )
        for block_id, block_value in enumerate(self._hash_blocks(recipe_card_hash)):
            self._db.execute(
                "INSERT OR IGNORE INTO simhash_index (block_id, block_value, hash) "
                "VALUES (?, ?, ?)",
                (block_id, block_value, recipe_card_hash),
            )
        self._db.commit()

    def update_status(
        self, recipe_card_hash: str, status: HashStatus,
        *,
        raw_text: str | None = None,
        model: str | None = None,
        output: str | None = None,
    ) -> None:
        set_clauses = ["status = ?"]
        params: list[str] = [status.value]

        if raw_text is not None:
            set_clauses.append("raw_text = ?")
            params.append(raw_text)
        if model is not None:
            set_clauses.append("model = ?")
            params.append(model)
        if output is not None:
            set_clauses.append("output = ?")
            params.append(output)

        params.append(recipe_card_hash)
        self._db.execute(
            f"UPDATE simhashes SET {', '.join(set_clauses)} WHERE hash = ?",
            params,
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
        """Strip noise that varies between duplicates of the same recipe.

        Removed before tokenization so these superficial differences
        don't affect the SimHash fingerprint:
            - URLs: CDN paths, tracking params, different site mirrors
            - Dates: publish/update timestamps differ, body is identical
            - Markdown formatting tokens: `` ` `` `*` `_` `>` `#` `[]` `()`
              differ between trafilatura / html2text / wprm_print output
        """
        text = text.lower()
        text = re.sub(r"https?://\S+", " ", text)
        text = re.sub(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}", " ", text)
        text = re.sub(r"[`*_>#\[\]()]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Split normalized text into meaningful content-bearing tokens.

        Token rules:
            - 2+ characters only: single-char tokens (punctuation, CJK
              particles like 的/了/は) are noise — they vote weakly and
              inconsistently, so we drop them.
            - \\w: ASCII word characters (ingredient, recipe, sauté, …)
            - ぀-ヿ: Japanese Hiragana + Katakana (レシピ, 調理, …)
            - 一-鿿: CJK Unified Ideographs (食材, 烹饪, 炒, …)

        Returns:
            Token list to be hashed for fingerprint building.
        """
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
