"""Data contracts for model export.

Used by ``BaseExporter`` → external consumers (deployment, serving).
"""

from __future__ import annotations

from dataclasses import dataclass

from labeling_sft.contracts.training import ArtifactLocation


@dataclass
class ExportResult:
    """Return value of ``BaseExporter.export()``."""

    artifact: ArtifactLocation
    format: str  # "merged_hf" | "gguf_q8_0" | "vllm_bf16" | ...
    size_mb: float  # total artifact size
    base_model_id: str
