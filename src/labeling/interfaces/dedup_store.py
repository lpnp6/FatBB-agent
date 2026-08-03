"""DedupStore — abstract interface for recipe-card content deduplication."""

from __future__ import annotations

from abc import ABC, abstractmethod
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


class DedupStore(ABC):
    """Persistent store of recipe-card content hashes with lifecycle tracking.

    Responsibilities:
        1. Sampling: group duplicate recipe files (same content, different
           filenames) so only one enters the work queue.
        2. Labeling gate: check each file's hash BEFORE calling the model.
           Hashes are registered in IN_FLIGHT state first, preventing
           concurrent duplicate submissions.
        3. Crash recovery: on restart, the checkpoint manager calls
           clear_in_flight_by_slugs() with slugs that were in-flight at
           crash time. Those hashes are removed so the files can be retried.
        4. Staleness: hung requests that never complete are expired via
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
        source_file: str,
        status: HashStatus,
        *,
        raw_text: str | None = None,
    ) -> None:
        """Persist a hash with its initial lifecycle status.

        Called with IN_FLIGHT before processing begins, then transitioned
        to ACCEPTED or REJECTED via update_status() once complete.

        Args:
            recipe_card_hash: Content fingerprint of the recipe card.
            source_file: Original file path or slug for provenance.
            status: Initial lifecycle state (almost always IN_FLIGHT).
            raw_text: The raw recipe markdown. Stored alongside the hash
                so the dedup store becomes the authoritative source of
                labeled documents — no external manifest needed.
        """
        ...

    @abstractmethod
    def update_status(
        self, recipe_card_hash: str, status: HashStatus,
        *,
        raw_text: str | None = None,
    ) -> None:
        """Transition an existing hash to a new status.

        Called after processing completes:
            - Success: update_status(hash, ACCEPTED)
            - Failure: update_status(hash, REJECTED)

        Args:
            recipe_card_hash: Content fingerprint to transition.
            status: Target lifecycle state.
            raw_text: If provided, store/update the raw markdown for this
                hash (typically passed when transitioning to ACCEPTED).
        """
        ...

    # ---- crash recovery ----------------------------------------------------

    @abstractmethod
    def clear_in_flight_by_slugs(self, slugs: set[str]) -> None:
        """Remove IN_FLIGHT entries for the given source_file slugs.

        Called once at startup, after the checkpoint manager identifies
        files that were IN_FLIGHT at crash time (now reset to pending).
        This ensures those files can be re-submitted on the next run.

        Args:
            slugs: Source file slugs that should no longer be IN_FLIGHT.
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
