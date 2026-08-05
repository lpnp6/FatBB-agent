"""Abstract storage boundary for model artifacts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from labeling_sft.contracts import ArtifactLocation


class BaseArtifactStore(ABC):
    @abstractmethod
    def materialize(self, source: ArtifactLocation) -> Path:
        """Make *source* available as a local path."""

    @abstractmethod
    def publish(self, source: Path, target: ArtifactLocation) -> ArtifactLocation:
        """Publish a local file or directory to *target*."""

    @abstractmethod
    def size_bytes(self, artifact: ArtifactLocation) -> int:
        """Return the size of *artifact* in bytes."""
