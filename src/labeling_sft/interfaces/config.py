"""Abstract base class for training / evaluation configuration."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Self


class BaseConfig(ABC):
    """Abstract configuration for training and evaluation.

    Subclasses may be dataclasses, pydantic models, or plain classes —
    they only need to satisfy the property and method contracts below.

    Concrete implementations:
        - :class:`~labeling_sft.configs.qlora.QLoRAConfig`
        - Future: ``FullFinetuneConfig``, ``DPOConfig``, ...
    """

    # ── Required properties ──────────────────────────────────────────────

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Hugging Face model ID or local path."""
        ...

    @property
    @abstractmethod
    def output_dir(self) -> str:
        """Model / adapter output directory."""
        ...

    @property
    @abstractmethod
    def seed(self) -> int:
        """Random seed for reproducibility."""
        ...

    # ── Serialization ────────────────────────────────────────────────────

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        ...

    @classmethod
    @abstractmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Deserialize from a dict."""
        ...

    @classmethod
    @abstractmethod
    def from_cli_args(cls, args: Any) -> Self:
        """Build from an ``argparse.Namespace``, overriding only explicitly-set fields.

        Typical implementation: instantiate defaults, then for each field
        use the *args* value if it differs from the default.
        """
        ...

    # ── Validation ───────────────────────────────────────────────────────

    def validate(self) -> list[str]:
        """Validate configuration.  Returns a list of problem descriptions.

        Empty list = valid.  Override in subclasses to add custom checks
        (e.g. ``lora_r > 0``, ``max_seq_length >= 512``).
        """
        return []
