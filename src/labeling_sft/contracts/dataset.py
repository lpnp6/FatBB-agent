"""Data contracts for dataset building and loading.

Used by ``BaseDatasetBuilder``, ``BaseTrainer``, and ``BaseEvaluator``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ── Data location abstraction ─────────────────────────────────────────────

class DataLocationType(str, Enum):
    """Supported data location backends."""

    LOCAL_PATH = "local_path"    # /data/train.jsonl
    HF_HUB = "hf_hub"            # username/dataset-name[:revision]
    S3_URI = "s3_uri"            # s3://bucket/prefix/train.parquet
    GCS_URI = "gcs_uri"          # gs://bucket/prefix/train.parquet
    IN_MEMORY = "in_memory"      # direct object reference (not serializable)


@dataclass
class DataLocation:
    """Abstract reference to a dataset location — not a concrete path.

    Consumers resolve this descriptor based on :attr:`type` to obtain the
    actual data (file handle, HuggingFace Dataset, Arrow Table, etc.).
    """

    type: DataLocationType
    value: str                   # path / URI / HF repo id
    version: str | None = None   # git commit / HF revision / object version
    metadata: dict[str, Any] = field(default_factory=dict)
    # split name, file glob, format hint, etc.

    # ── Convenience constructors ─────────────────────────────────────────

    @classmethod
    def local(cls, path: str, **meta: Any) -> "DataLocation":
        """Local filesystem path (single file or directory)."""
        return cls(type=DataLocationType.LOCAL_PATH, value=path, metadata=meta)

    @classmethod
    def hf_hub(cls, repo: str, *, revision: str | None = None, **meta: Any) -> "DataLocation":
        """HuggingFace Hub dataset repository."""
        return cls(type=DataLocationType.HF_HUB, value=repo, version=revision, metadata=meta)

    @classmethod
    def s3(cls, uri: str, *, version: str | None = None, **meta: Any) -> "DataLocation":
        """S3 object URI."""
        return cls(type=DataLocationType.S3_URI, value=uri, version=version, metadata=meta)

    @classmethod
    def gcs(cls, uri: str, *, version: str | None = None, **meta: Any) -> "DataLocation":
        """GCS object URI."""
        return cls(type=DataLocationType.GCS_URI, value=uri, version=version, metadata=meta)


# ── Dataset records and split ─────────────────────────────────────────────

@dataclass
class DatasetRecord:
    """Single Alpaca-format sample."""

    instruction: str
    input: str  # raw markdown document
    output: str  # compact JSON string (gold label)


@dataclass
class DatasetStats:
    """Dataset split statistics."""

    total_valid_records: int
    skipped_records: int
    recipe_count: int
    not_a_recipe_count: int
    train_count: int
    val_count: int
    val_split: float
    seed: int
    domain_distribution: dict[str, dict[str, int]] = field(default_factory=dict)
    # Example: {"recipetineats": {"train": 183, "val": 32}, ...}


@dataclass(frozen=True)
class DatasetBuildRequest:
    """Input contract for a dataset build operation.

    ``source`` describes the labeled input and the ``*_target`` fields
    describe where the builder should persist its artifacts. A concrete
    builder declares which :class:`DataLocationType` values it supports.
    """

    source: DataLocation
    train_target: DataLocation
    val_target: DataLocation
    stats_target: DataLocation | None = None
    val_split: float = 0.15
    seed: int = 42


@dataclass
class DatasetSplit:
    """Return value of ``BaseDatasetBuilder.build()``.

    Carries abstract :class:`DataLocation` references rather than raw paths
    so that consumers can resolve data from any supported backend.

    Examples::

        # Local files
        DatasetSplit(
            train=DataLocation.local("/data/train.jsonl"),
            val=DataLocation.local("/data/val.jsonl"),
            stats=...,
        )

        # HuggingFace Hub
        DatasetSplit(
            train=DataLocation.hf_hub("fatbb/recipe-labels", revision="v1"),
            val=DataLocation.hf_hub("fatbb/recipe-labels", revision="v1"),
            stats=...,
        )
    """

    train: DataLocation
    val: DataLocation
    stats: DatasetStats | None = None  # optional when consuming pre-built data
