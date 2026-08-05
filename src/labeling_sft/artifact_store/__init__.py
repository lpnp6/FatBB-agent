"""Artifact-store implementations and location dispatch."""

from labeling_sft.contracts import ArtifactLocation, ArtifactLocationType
from labeling_sft.artifact_store.local import LocalArtifactStore
from labeling_sft.interfaces.artifact_store import BaseArtifactStore

__all__ = ["BaseArtifactStore", "LocalArtifactStore", "artifact_store"]


def artifact_store(location: ArtifactLocation) -> BaseArtifactStore:
    """Return the storage backend for an artifact location."""
    if location.type is ArtifactLocationType.LOCAL_PATH:
        return LocalArtifactStore()
    raise NotImplementedError(f"No artifact store configured for {location.type.value}")
