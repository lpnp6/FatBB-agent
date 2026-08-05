"""Abstract base class for model exporters."""

from __future__ import annotations

from abc import ABC, abstractmethod

from labeling_sft.contracts import ArtifactLocation, ExportResult, TrainingResult


class BaseExporter(ABC):
    """Convert a trained adapter / checkpoint into a deployable format.

    Each export format is a separate ``BaseExporter`` subclass —
    they share no implementation, only the contract.

    Concrete implementations:
        - :class:`~labeling_sft.exporters.gguf.GGUFExporter` (format ``"gguf_q8_0"``)
        - Future: ``VLLMExporter``, ``OllamaExporter``, ...
    """

    @abstractmethod
    def export(
        self,
        training: TrainingResult,
        target: ArtifactLocation,
        **kwargs,
    ) -> ExportResult:
        """Execute model export.

        Args:
            training: Trained model artifacts and base model identity.
            target: Export target location.
            **kwargs:      Exporter-specific options.

        Returns:
            :class:`~labeling_sft.interfaces.contracts.ExportResult`.

        Raises:
            FileNotFoundError: A required artifact does not exist.
        """
        ...

    @property
    @abstractmethod
    def format_name(self) -> str:
        """Format identifier for this exporter.

        Examples: ``"gguf_q8_0"``, ``"vllm_bf16"``.
        Used by CLI and auto-discovery registries.
        """
        ...
