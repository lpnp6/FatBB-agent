"""Local filesystem artifact storage."""

from __future__ import annotations

import shutil
from pathlib import Path

from labeling_sft.contracts import ArtifactLocation
from labeling_sft.interfaces.artifact_store import BaseArtifactStore


class LocalArtifactStore(BaseArtifactStore):
    def materialize(self, source: ArtifactLocation) -> Path:
        path = Path(source.value)
        if not path.exists():
            raise FileNotFoundError(f"Artifact not found: {path}")
        return path

    def publish(self, source: Path, target: ArtifactLocation) -> ArtifactLocation:
        destination = Path(target.value)
        if destination.is_dir():
            destination /= source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        return ArtifactLocation.local(str(destination.resolve()), **target.metadata)

    def size_bytes(self, artifact: ArtifactLocation) -> int:
        return Path(artifact.value).stat().st_size
