"""Merge a QLoRA adapter into its base model and export the full model.

Loads the base model in bfloat16 (no quantisation — needed for a clean merge),
attaches the LoRA adapter, calls ``merge_and_unload()``, and saves the result
as a standard Hugging Face model directory.  The output can be served with
vLLM, loaded by ``transformers``, or converted to GGUF for ollama.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from labeling_sft.contracts import ExportResult
from labeling_sft.interfaces.exporter import BaseExporter

logger = logging.getLogger(__name__)


class MergeExporter(BaseExporter):
    """Merge a LoRA adapter into the base model, producing a full HF model directory.

    Usage::

        exporter = MergeExporter()
        result = exporter.export(
            adapter_dir="models/qwen2.5-3b-fatbb-v1",
            output_dir="models/qwen2.5-3b-fatbb-v1-merged",
        )
    """

    @property
    def format_name(self) -> str:
        return "merged_hf"

    def export(
        self,
        adapter_dir: str,
        output_dir: str,
        base_model_id: str | None = None,
        **kwargs,
    ) -> ExportResult:
        """Merge adapter → full model and save."""
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        adapter_path = Path(adapter_dir)
        if not adapter_path.is_dir():
            raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")

        base = base_model_id or "Qwen/Qwen2.5-3B-Instruct"

        output_path = Path(output_dir)
        if output_path.exists():
            logger.warning("Output directory exists, removing: %s", output_path)
            shutil.rmtree(output_path)

        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        logger.info("Loading base model in %s (no quantisation for clean merge)", str(compute_dtype))

        model = AutoModelForCausalLM.from_pretrained(
            base,
            torch_dtype=compute_dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True)

        logger.info("Loading LoRA adapter: %s", adapter_dir)
        model = PeftModel.from_pretrained(model, adapter_dir)

        logger.info("Merging adapter into base model ...")
        merged = model.merge_and_unload()

        logger.info("Saving merged model to %s", output_path)
        merged.save_pretrained(output_path, safe_serialization=True)
        tokenizer.save_pretrained(output_path)

        size_mb = sum(
            f.stat().st_size for f in output_path.rglob("*") if f.is_file()
        ) / (1024 * 1024)
        logger.info("Export complete — %.0f MB written to %s", size_mb, output_path)

        return ExportResult(
            output_path=str(output_path.resolve()),
            format=self.format_name,
            size_mb=round(size_mb, 1),
            base_model_id=base,
        )
