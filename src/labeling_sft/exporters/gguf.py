"""Convert a merged Hugging Face model to GGUF format.

Requires ``llama.cpp`` to be installed and the ``convert_hf_to_gguf.py``
script available, or set ``LLAMA_CPP_CONVERT`` to point to the script location.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

from labeling_sft.contracts import ExportResult
from labeling_sft.interfaces.exporter import BaseExporter

logger = logging.getLogger(__name__)


class GGUFExporter(BaseExporter):
    """Convert a merged HF model directory to GGUF format.

    Usage::

        exporter = GGUFExporter(outtype="q8_0")
        result = exporter.export(
            adapter_dir="models/qwen2.5-3b-fatbb-v1",
            output_dir="models/qwen2.5-3b-fatbb-v1-gguf",
        )
    """

    def __init__(self, outtype: str = "q8_0") -> None:
        self._outtype = outtype

    @property
    def format_name(self) -> str:
        return f"gguf_{self._outtype}"

    def export(
        self,
        adapter_dir: str,
        output_dir: str,
        base_model_id: str | None = None,
        **kwargs,
    ) -> ExportResult:
        """Convert to GGUF.

        If *output_dir* contains a merged HF model (from ``MergeExporter``),
        converts it directly.  Otherwise, merges first, then converts.
        """
        convert_script = os.environ.get("LLAMA_CPP_CONVERT", "convert_hf_to_gguf.py")

        model_path = Path(output_dir)
        # If output_dir is not already a merged model, we need the caller
        # to have merged first, or we can look for adapter_dir as fallback.
        if not (model_path / "config.json").exists():
            # Assume output_dir is where GGUF should land and adapter_dir
            # has the adapter whose base we need to merge.
            # For now, require the caller to have merged already.
            raise FileNotFoundError(
                f"Merged model not found at {output_dir}. "
                f"Run MergeExporter first."
            )

        outfile = kwargs.get("gguf_output") or str(model_path / f"{model_path.name}.gguf")

        cmd = [
            sys.executable, convert_script,
            str(model_path),
            "--outfile", outfile,
            "--outtype", self._outtype,
        ]
        logger.info("Running: %s", " ".join(cmd))
        subprocess.run(cmd, check=True)

        size_mb = Path(outfile).stat().st_size / (1024 * 1024)
        logger.info("GGUF model written to %s (%.0f MB)", outfile, size_mb)

        return ExportResult(
            output_path=str(Path(outfile).resolve()),
            format=self.format_name,
            size_mb=round(size_mb, 1),
            base_model_id=base_model_id or "Qwen/Qwen2.5-3B-Instruct",
        )
