"""Data contracts for model training.

Used by ``BaseTrainer`` → ``BaseExporter`` / ``BaseEvaluator``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ArtifactLocationType(str, Enum):
    """Supported backends for model artifacts."""

    LOCAL_PATH = "local_path"
    HF_HUB = "hf_hub"
    S3_URI = "s3_uri"
    GCS_URI = "gcs_uri"


@dataclass(frozen=True)
class ArtifactLocation:
    """Reference to a model artifact, independent of its storage backend."""

    type: ArtifactLocationType
    value: str
    version: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def local(cls, path: str, **meta: Any) -> "ArtifactLocation":
        """Artifact stored on the local filesystem."""
        return cls(type=ArtifactLocationType.LOCAL_PATH, value=path, metadata=meta)

    @classmethod
    def hf_hub(
        cls, repo: str, *, revision: str | None = None, **meta: Any
    ) -> "ArtifactLocation":
        """Artifact stored in a Hugging Face Hub repository."""
        return cls(
            type=ArtifactLocationType.HF_HUB,
            value=repo,
            version=revision,
            metadata=meta,
        )

    @classmethod
    def s3(
        cls, uri: str, *, version: str | None = None, **meta: Any
    ) -> "ArtifactLocation":
        """Artifact stored in S3."""
        return cls(
            type=ArtifactLocationType.S3_URI,
            value=uri,
            version=version,
            metadata=meta,
        )

    @classmethod
    def gcs(
        cls, uri: str, *, version: str | None = None, **meta: Any
    ) -> "ArtifactLocation":
        """Artifact stored in Google Cloud Storage."""
        return cls(
            type=ArtifactLocationType.GCS_URI,
            value=uri,
            version=version,
            metadata=meta,
        )


@dataclass
class TrainingResult:
    """Return value of ``BaseTrainer.train()``.

    Exporter and Evaluator consume these abstract artifact locations; they do
    not need to know how training was performed or where artifacts are stored.
    """

    model: ArtifactLocation
    adapter: ArtifactLocation  # for full FT, typically the same as ``model``
    base_model_id: str  # HuggingFace base model ID
    final_eval_loss: float | None = None
    total_steps: int = 0
    best_checkpoint: ArtifactLocation | None = None
