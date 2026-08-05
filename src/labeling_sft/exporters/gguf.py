"""Merge a trained LoRA adapter and convert it to GGUF."""

from __future__ import annotations

import logging
import subprocess
import sys
from tempfile import TemporaryDirectory
from pathlib import Path

from labeling_sft.contracts import ArtifactLocation, ExportResult, TrainingResult
from labeling_sft.artifact_store import artifact_store
from labeling_sft.interfaces.exporter import BaseExporter

logger = logging.getLogger(__name__)


class GGUFExporter(BaseExporter):
    """Merge a training adapter and publish a GGUF model.

    Usage::

        exporter = GGUFExporter(outtype="q8_0")
        result = exporter.export(
            training=training_result,
            target=ArtifactLocation.local("models/qwen2.5-3b-fatbb-v1.gguf"),
        )
    """

    def __init__(self, outtype: str = "q8_0") -> None:
        self._outtype = outtype

    @property
    def format_name(self) -> str:
        return f"gguf_{self._outtype}"

    def export(
        self,
        training: TrainingResult,
        target: ArtifactLocation,
        **kwargs,
    ) -> ExportResult:
        """Merge the adapter, convert it, and publish the GGUF artifact."""
        import torch # pyright: ignore[reportMissingImports]
        from peft import PeftModel # pyright: ignore[reportMissingImports]
        from transformers import AutoModelForCausalLM, AutoTokenizer # pyright: ignore[reportMissingImports]

        adapter_path = artifact_store(training.adapter).materialize(training.adapter)
        target_store = artifact_store(target)
        convert_script = self._convert_script()
        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        with TemporaryDirectory(prefix="fatbb-gguf-") as temporary:
            work_dir = Path(temporary)
            merged_path = work_dir / "merged"
            gguf_path = work_dir / "model.gguf"

            logger.info("Loading base model in %s", compute_dtype)
            model = AutoModelForCausalLM.from_pretrained(
                training.base_model_id,
                torch_dtype=compute_dtype,
                device_map="auto",
                trust_remote_code=True,
            )
            tokenizer = AutoTokenizer.from_pretrained(training.base_model_id, trust_remote_code=True)
            model = PeftModel.from_pretrained(model, adapter_path).merge_and_unload()
            model.save_pretrained(merged_path, safe_serialization=True)
            tokenizer.save_pretrained(merged_path)

            cmd = [
                sys.executable, str(convert_script), str(merged_path),
                "--outfile", str(gguf_path), "--outtype", self._outtype,
            ]
            logger.info("Running: %s", " ".join(cmd))
            subprocess.run(cmd, check=True)
            artifact = target_store.publish(gguf_path, target)

        size_mb = target_store.size_bytes(artifact) / (1024 * 1024)
        logger.info("GGUF model written to %s (%.0f MB)", artifact.value, size_mb)

        return ExportResult(
            artifact=artifact,
            format=self.format_name,
            size_mb=round(size_mb, 1),
            base_model_id=training.base_model_id,
        )

    @staticmethod
    def _convert_script() -> Path:
        tool_dir = Path.home() / ".fatbb" / "tools" / "llama.cpp"
        script = tool_dir / "convert_hf_to_gguf.py"
        if script.is_file():
            return script
        if tool_dir.exists():
            raise FileNotFoundError(f"llama.cpp converter not found in {tool_dir}")

        tool_dir.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Installing llama.cpp converter in %s", tool_dir)
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "https://github.com/ggml-org/llama.cpp.git",
                str(tool_dir),
            ],
            check=True,
        )
        if not script.is_file():
            raise FileNotFoundError(f"llama.cpp converter not found after install: {script}")
        return script
