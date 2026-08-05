"""Backward-compatible re-export from the new ``configs/`` sub-package.

.. deprecated:: transitional
    Import from :mod:`labeling_sft.configs` directly in new code.
"""

from __future__ import annotations

from labeling_sft.configs.qlora import QLoRAConfig, _qlora_config_fields as _config_fields  # noqa: F401
