"""DedupStore — abstract interface for recipe-card content deduplication."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum


class HashStatus(str, Enum):
    """Lifecycle of a recipe-card hash in the dedup store."""

    IN_FLIGHT = "in_flight"
    """Hash registered BEFORE API call. Prevents concurrent duplicate submits."""

    ACCEPTED = "accepted"
    """API call succeeded + validated. Recipe is fully labeled."""

    REJECTED = "rejected"
    """API call failed, or file was classified as non-recipe / invalid.

    Blocked from future calls so the same bad file is not re-submitted.
    """


class DedupStore(ABC):
    """Persistent store of recipe-card content hashes with lifecycle tracking.

    Responsibilities:
        1. Sampling: group duplicate recipe files (wprm_print vs full_page
           variants of the same recipe card).
        2. Labeling gate: check each file's hash BEFORE calling the model.
           Hashes are registered in IN_FLIGHT state first, preventing
           concurrent duplicate submissions.
        3. Crash recovery: on restart, the checkpoint manager calls
           clear_in_flight_by_slugs() with slugs that were in-flight at
           crash time. Those hashes are removed so the files can be retried.

    Two planned implementations:
        - SQLiteDedupStore  (persistent, survives process restarts)
        - MemoryDedupStore  (in-memory set, for unit tests, lost on restart)
    """

    # ---- lifecycle queries -------------------------------------------------

    @abstractmethod
    def lookup(self, recipe_card_hash: str) -> HashStatus | None:
        """Return the current status of a hash, or None if unknown.

        Called before every client.label() call:
            - ACCEPTED / REJECTED -> skip, no API cost
            - IN_FLIGHT -> skip, another variant is being labeled right now
            - None -> proceed to register(IN_FLIGHT) then call API
        """
        ...

    # ---- registration ------------------------------------------------------

    @abstractmethod
    def register(
        self,
        recipe_card_hash: str,
        source_file: str,
        status: HashStatus,
    ) -> None:
        """Persist a hash with its initial lifecycle status.

        Called:
            - Before API call: register(hash, slug, IN_FLIGHT)
            - During sampling: register(hash, slug, ACCEPTED) for the
              winning variant after dedup resolution.

        Args:
            recipe_card_hash: SHA-256 fingerprint of the recipe card.
            source_file: Original file path or slug for provenance.
            status: Initial lifecycle state.
        """
        ...

    @abstractmethod
    def update_status(
        self, recipe_card_hash: str, status: HashStatus
    ) -> None:
        """Transition an existing hash to a new status.

        Called after API call completes:
            - Success: update_status(hash, ACCEPTED)
            - Failure: update_status(hash, REJECTED)
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
        """Compute a stable SHA-256 fingerprint of the recipe card.

        Extracts the ### Ingredients + ### Instructions blocks from the
        markdown, normalizes whitespace, then SHA-256 hashes the result.
        This is deliberately NOT a semantic hash — it only matches
        byte-identical recipe cards (same recipe, same site, different URL
        variant). For fuzzy matching, swap the implementation.

        Args:
            markdown: Full recipe Markdown text (may include intro, comments).

        Returns:
            64-character hex SHA-256 digest.
        """
        ...
