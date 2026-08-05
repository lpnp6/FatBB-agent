"""Abstract base class for model exporters."""

from __future__ import annotations

from abc import ABC, abstractmethod

from labeling_sft.contracts import ExportResult


class BaseExporter(ABC):
    """Convert a trained adapter / checkpoint into a deployable format.

    Each export format is a separate ``BaseExporter`` subclass —
    they share no implementation, only the contract.

    Concrete implementations:
        - :class:`~labeling_sft.exporters.merge.MergeExporter` (format ``"merged_hf"``)
        - :class:`~labeling_sft.exporters.gguf.GGUFExporter` (format ``"gguf_q8_0"``)
        - Future: ``VLLMExporter``, ``OllamaExporter``, ...
    """

    @abstractmethod
    def export(
        self,
        adapter_dir: str,
        output_dir: str,
        base_model_id: str | None = None,
        **kwargs,
    ) -> ExportResult:
        """Execute model export.

        Args:
            adapter_dir:   Trained adapter / checkpoint directory.
            output_dir:    Export target directory.
            base_model_id: Base model HuggingFace ID (``None`` = auto-detect).
            **kwargs:      Exporter-specific options.

        Returns:
            :class:`~labeling_sft.interfaces.contracts.ExportResult`.

        Raises:
            FileNotFoundError: ``adapter_dir`` does not exist.
        """
        ...

    @property
    @abstractmethod
    def format_name(self) -> str:
        """Format identifier for this exporter.

        Examples: ``"merged_hf"``, ``"gguf_q8_0"``, ``"vllm_bf16"``.
        Used by CLI and auto-discovery registries.
        """
        ...
