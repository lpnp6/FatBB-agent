"""Terminal-independent events accepted by the UI reducer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


# Keys understood by the UI state machine. Keeping this as a shared alias
# prevents callers from widening the type to arbitrary terminal strings.
Key = Literal["up", "down", "enter", "escape", "ctrl_d"]


@dataclass(frozen=True)
class InputChanged:
    text: str


@dataclass(frozen=True)
class KeyPressed:
    key: Key
