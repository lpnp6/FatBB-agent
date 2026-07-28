"""Terminal-independent events accepted by the UI reducer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class InputChanged:
    text: str


@dataclass(frozen=True)
class KeyPressed:
    key: Literal["up", "down", "enter", "escape", "ctrl_d"]
