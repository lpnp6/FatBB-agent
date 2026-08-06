"""Merge a trained LoRA adapter and convert it to GGUF."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from tempfile import TemporaryDirectory
from pathlib import Path

from labeling_sft.contracts import ArtifactLocation, ExportResult, TrainingResult
from labeling_sft.artifact_store import artifact_store
from labeling_sft.interfaces.exporter import BaseExporter

logger = logging.getLogger(__name__)


def training_result_from_checkpoint(checkpoint: str | Path) -> TrainingResult:
    """Build a training result from a local PEFT checkpoint directory."""
    path = Path(checkpoint).expanduser().resolve()
    config_path = path / "adapter_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Adapter config not found: {config_path}")
    if not any((path / name).is_file() for name in ("adapter_model.safetensors", "adapter_model.bin")):
        raise FileNotFoundError(f"Adapter weights not found in: {path}")

    adapter_config = json.loads(config_path.read_text(encoding="utf-8"))
    base_model_id = adapter_config.get("base_model_name_or_path")
    if not isinstance(base_model_id, str) or not base_model_id:
        raise ValueError(f"Missing base_model_name_or_path in: {config_path}")

    state_path = path / "trainer_state.json"
    total_steps = 0
    if state_path.is_file():
        total_steps = int(json.loads(state_path.read_text(encoding="utf-8")).get("global_step", 0))
    artifact = ArtifactLocation.local(str(path))
    return TrainingResult(
        model=artifact,
        adapter=artifact,
        base_model_id=base_model_id,
        total_steps=total_steps,
    )


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

        # Prefer D: on Windows — the system temp dir on C: is often too small
        _tmp_root = None
        if sys.platform == "win32":
            for _drive in ("D:", "E:", "F:"):
                _candidate = Path(_drive) / "fatbb_tmp"
                try:
                    _candidate.mkdir(parents=True, exist_ok=True)
                    _tmp_root = str(_candidate)
                    break
                except OSError:
                    continue
        with TemporaryDirectory(prefix="fatbb-gguf-", dir=_tmp_root) as temporary:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a PEFT checkpoint directory to GGUF")
    parser.add_argument("--checkpoint", "--checkpoint-dir", dest="checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="Output GGUF file")
    parser.add_argument("--outtype", default="q8_0", help="GGUF quantization type")
    args = parser.parse_args()

    result = GGUFExporter(args.outtype).export(
        training_result_from_checkpoint(args.checkpoint),
        ArtifactLocation.local(str(args.output.expanduser().resolve())),
    )
    print(json.dumps({"artifact": result.artifact.value, "size_mb": result.size_mb}))


if __name__ == "__main__":
    main()
