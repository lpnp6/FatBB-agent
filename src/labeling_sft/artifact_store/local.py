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
        if destination.is_dir() and source.is_file():
            destination /= source.name
        if source.resolve() == destination.resolve():
            return ArtifactLocation.local(str(destination), **target.metadata)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(source, destination)
        return ArtifactLocation.local(str(destination.resolve()), **target.metadata)

    def size_bytes(self, artifact: ArtifactLocation) -> int:
        path = Path(artifact.value)
        if path.is_file():
            return path.stat().st_size
        return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())
