"""DedupStore — abstract interface for recipe-card content deduplication."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .checkpoint_store import CheckpointStore


class HashStatus(str, Enum):
    """Lifecycle of a recipe-card hash in the dedup store."""

    IN_FLIGHT = "in_flight"
    """Hash registered BEFORE processing. Prevents concurrent duplicate submits."""

    ACCEPTED = "accepted"
    """Processing succeeded + validated. Recipe is fully labeled."""

    REJECTED = "rejected"
    """Processing failed, or file was classified as non-recipe / invalid.

    Blocked from future attempts so the same bad file is not re-submitted.
    """


@dataclass
class DedupEntry:
    """A single entry for batch registration in the dedup store."""

    recipe_card_hash: str
    status: HashStatus
    source_id: str | None = None
    raw_text: str | None = None
    model: str | None = None
    output: str | None = None
    _extra: dict[str, object] = field(default_factory=dict, repr=False)


class DedupStore(ABC):
    """Persistent store of recipe-card content hashes with lifecycle tracking.

    Responsibilities:
        1. Content storage: persist raw recipe markdown alongside its
           fingerprint, making the dedup store the authoritative source
           of labeled documents.
        2. Sampling: group duplicate recipe files (same content, different
           filenames) so only one enters the work queue.
        3. Labeling gate: check each file's hash BEFORE calling the model.
           Hashes are registered in IN_FLIGHT state first, preventing
           concurrent duplicate submissions.
        4. Crash recovery: the embedded CheckpointStore tracks per-item
           state. On restart, IN_FLIGHT items are reset and retried.
        5. Staleness: hung requests that never complete are expired via
           expire_stale() so their duplicates can be retried.

    Two planned implementations:
        - SQLiteDedupStore  (persistent, survives process restarts)
        - MemoryDedupStore  (in-memory set, for unit tests, lost on restart)

    Every concrete implementation embeds a CheckpointStore to track per-item
    progress through the labeling pipeline (PENDING → IN_FLIGHT →
    COMPLETED / REJECTED).
    """

    # ---- embedded store -----------------------------------------------------

    @property
    @abstractmethod
    def checkpoint(self) -> CheckpointStore:
        """The CheckpointStore tracking per-item lifecycle for this dedup store.

        Concrete implementations receive a CheckpointStore at construction
        time and expose it here. The orchestrator uses this to delegate
        per-item state transitions without managing two separate stores.
        """
        ...

    # ---- factory -------------------------------------------------------------

    @abstractmethod
    def create_in_memory(self) -> DedupStore:
        """Return a new, empty instance of the same dedup store type.

        Used by the sampler for in-memory near-duplicate clustering among
        freshly discovered files — without polluting the persistent store.

        The returned instance must share the same dedup algorithm and
        threshold as ``self``.
        """
        ...

    # ---- lifecycle queries -------------------------------------------------

    @abstractmethod
    def lookup(self, recipe_card_hash: str) -> HashStatus | None:
        """Return the current status of a hash, or None if unknown.

        Called before processing each file:
            - ACCEPTED -> skip, already completed
            - REJECTED -> skip, known bad file
            - IN_FLIGHT -> skip, another copy is being processed right now
              (the duplicate file should be put back into the pending queue
              so it can be retried if the in-flight request ultimately fails)
            - None -> proceed to register(IN_FLIGHT) then process

        Implementations should treat stale IN_FLIGHT entries (registered
        longer ago than the configured timeout) as expired — returning None
        so the file can be retried.
        """
        ...

    # ---- staleness ---------------------------------------------------------

    @abstractmethod
    def expire_stale(self, timeout_minutes: int) -> int:
        """Remove IN_FLIGHT entries older than the given timeout.

        Called at startup and periodically during long runs to prevent
        hung requests from permanently blocking their duplicates.

        Args:
            timeout_minutes: Entries in IN_FLIGHT longer than this are removed.

        Returns:
            Number of entries expired.
        """
        ...

    # ---- registration ------------------------------------------------------

    @abstractmethod
    def register(
        self,
        recipe_card_hash: str,
        status: HashStatus,
        *,
        source_id: str | None = None,
        raw_text: str | None = None,
        model: str | None = None,
        output: str | None = None,
    ) -> None:
        """Persist a hash with its initial lifecycle status.

        Called with IN_FLIGHT before processing begins, then transitioned
        to ACCEPTED or REJECTED via update_status() once complete.

        Args:
            recipe_card_hash: Content fingerprint of the recipe card.
            status: Initial lifecycle state (almost always IN_FLIGHT).
            source_id: Stable unique identifier for the source document,
                produced by a URIResolver. Links the dedup entry back to
                its originating file/URI.
            raw_text: The raw recipe markdown.
            model: Model identifier that produced the labeling output.
            output: The labeling result JSON (only for ACCEPTED records).
        """
        ...

    @abstractmethod
    def update_status(
        self, recipe_card_hash: str, status: HashStatus,
        *,
        source_id: str | None = None,
        raw_text: str | None = None,
        model: str | None = None,
        output: str | None = None,
    ) -> None:
        """Transition an existing hash to a new status.

        Called after processing completes:
            - Success: update_status(hash, ACCEPTED, raw_text=..., model=..., output=...)
            - Failure: update_status(hash, REJECTED)

        Args:
            recipe_card_hash: Content fingerprint to transition.
            status: Target lifecycle state.
            source_id: If provided, store/update the source document identifier.
            raw_text: If provided, store/update the raw markdown.
            model: Model identifier, stored for provenance.
            output: Labeling result JSON, stored for training data.
        """
        ...

    # ---- batch operations ----------------------------------------------------

    @abstractmethod
    def lookup_batch(self, hashes: list[str]) -> dict[str, HashStatus | None]:
        """Return the current status for every hash in *hashes*.

        One query instead of N individual ``lookup()`` calls.  Keys in the
        returned dict are exactly the input hashes; values are ``None`` for
        hashes not yet registered.

        Args:
            hashes: Content fingerprints to query.

        Returns:
            Mapping of ``hash -> HashStatus | None``.
        """
        ...

    @abstractmethod
    def update_status_batch(self, entries: list[DedupEntry]) -> None:
        """Batch-update status and provenance for multiple hashes in one transaction.

        Each *DedupEntry* specifies ``recipe_card_hash``, ``status``, and
        optional ``source_id``, ``raw_text``, ``model``, ``output``.  Unlike
        :meth:`register_batch`, this does **UPDATE** — every hash must
        already exist in the store (inserted via :meth:`register` or
        :meth:`register_batch`).

        Args:
            entries: Updates to apply.  Fields set to ``None`` are skipped
                (the existing column value is left unchanged).
        """
        ...

    @abstractmethod
    def register_batch(self, entries: list[DedupEntry]) -> None:
        """Persist multiple hashes in a single transaction.

        Semantically equivalent to calling ``register()`` for each entry, but
        implementations should batch-write for throughput.  All entries are
        written atomically — partial failure rolls back.

        Args:
            entries: Entries to register together.
        """
        ...

    # ---- fingerprint computation -------------------------------------------

    @abstractmethod
    def recipe_card_hash(self, markdown: str) -> str:
        """Compute a stable fingerprint of the recipe card.

        The concrete implementation decides the algorithm:
            - Exact hash → catches byte-identical content across filenames
            - SimHash → fuzzy match, catches cross-site syndication
            - Ingredient-only hash → catches same dish in different languages

        Args:
            markdown: Full recipe Markdown text (may include intro, comments).

        Returns:
            String fingerprint. Format depends on implementation
            (hex digest, SimHash bitstring, etc.).
        """
        ...
