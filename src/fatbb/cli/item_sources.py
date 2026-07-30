"""CLI item-source handlers referenced by the source-controlled UI config.

They request presentation data through ``KnowledgeBaseService`` and never
import KB adapters or read ``kb.toml`` directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .controller import CliController


def existing_knowledge_bases(controller: CliController) -> tuple[str, ...]:
    return (*controller._existing_knowledge_base_items(), "Back")
