"""Merge a QLoRA adapter into its base model and export the full model.

Loads the base model in bfloat16 (no quantisation — needed for a clean merge),
attaches the LoRA adapter, calls ``merge_and_unload()``, and saves the result
as a standard Hugging Face model directory.  The output can be served with
vLLM, loaded by ``transformers``, or converted to GGUF for ollama.

Usage::

    PYTHONPATH=src python -m labeling_sft.export \\
        --adapter_dir models/qwen2.5-3b-fatbb-v1 \\
        --output_dir models/qwen2.5-3b-fatbb-v1-merged

    # With GGUF conversion (requires llama.cpp Python bindings):
    PYTHONPATH=src python -m labeling_sft.export \\
        --adapter_dir models/qwen2.5-3b-fatbb-v1 \\
        --output_dir models/qwen2.5-3b-fatbb-v1-merged \\
        --gguf
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def merge_and_save(
    adapter_dir: str,
    output_dir: str,
    base_model_id: str = "Qwen/Qwen2.5-3B-Instruct",
) -> str:
    """Merge the LoRA adapter into the base model and save to *output_dir*.

    Returns the resolved output directory path.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter_path = Path(adapter_dir)
    if not adapter_path.is_dir():
        raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")

    output_path = Path(output_dir)
    if output_path.exists():
        logger.warning("Output directory exists, removing: %s", output_path)
        shutil.rmtree(output_path)

    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    logger.info("Loading base model in %s (no quantisation for clean merge)", str(compute_dtype))

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=compute_dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)

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

    return str(output_path)


# ---------------------------------------------------------------------------
# GGUF conversion (optional)
# ---------------------------------------------------------------------------

def convert_to_gguf(model_dir: str, gguf_output: str | None = None) -> str:
    """Convert a merged Hugging Face model to GGUF format.

    Requires ``llama.cpp`` to be installed and the ``convert_hf_to_gguf.py``
    script available on ``PATH``, or set ``LLAMA_CPP_PYTHON`` to point to the
    script location.
    """
    import os
    import subprocess
    import sys

    convert_script = os.environ.get("LLAMA_CPP_CONVERT", "convert_hf_to_gguf.py")

    model_path = Path(model_dir)
    outfile = gguf_output or str(model_path / f"{model_path.name}.gguf")

    cmd = [
        sys.executable, convert_script,
        str(model_path),
        "--outfile", outfile,
        "--outtype", "q8_0",
    ]
    logger.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)

    logger.info("GGUF model written to %s", outfile)
    return outfile


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Merge QLoRA adapter into base model and export",
    )
    parser.add_argument("--adapter_dir", required=True,
                        help="Path to the saved LoRA adapter directory")
    parser.add_argument("--output_dir", required=True,
                        help="Directory to save the merged model")
    parser.add_argument("--base_model_id", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--gguf", action="store_true",
                        help="Convert merged model to GGUF format")
    parser.add_argument("--gguf_output", default=None,
                        help="GGUF output file path (default: <output_dir>/<name>.gguf)")
    args = parser.parse_args()

    merged_dir = merge_and_save(
        adapter_dir=args.adapter_dir,
        output_dir=args.output_dir,
        base_model_id=args.base_model_id,
    )

    if args.gguf:
        convert_to_gguf(merged_dir, args.gguf_output)
