"""Shared data types used across all labeling modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TokenUsage:
    """Token consumption for a single labeling call."""

    input: int = 0
    output: int = 0


@dataclass
class ExtractionResult:
    """Raw result from a LabelingClient call.

    This is the unvalidated, unparsed output straight from the model.
    Validation and parsing happen downstream in validator.py.
    """

    raw_output: str
    """Raw text response from the model (expected to be JSON)."""

    model: str
    """Identifier of the model that produced this result."""

    token_usage: TokenUsage = field(default_factory=TokenUsage)
    """Token consumption for this call."""

    latency_ms: int = 0
    """Wall-clock latency of the API call or local inference."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Backend-specific extras (API endpoint, finish_reason, etc.)."""
