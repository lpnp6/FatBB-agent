"""Backward-compatible re-export from the new ``exporters/`` sub-package.

.. deprecated:: transitional
    Import from :mod:`labeling_sft.exporters` directly in new code.
    Use :class:`MergeExporter` / :class:`GGUFExporter` instead of the bare
    ``merge_and_save()`` / ``convert_to_gguf()`` functions.
"""

from __future__ import annotations

from labeling_sft.exporters.gguf import GGUFExporter
from labeling_sft.exporters.merge import MergeExporter

__all__ = [
    "MergeExporter",
    "GGUFExporter",
    "merge_and_save",
    "convert_to_gguf",
]


def merge_and_save(
    adapter_dir: str,
    output_dir: str,
    base_model_id: str = "Qwen/Qwen2.5-3B-Instruct",
) -> str:
    """Merge the LoRA adapter into the base model and save to *output_dir*.

    .. deprecated::
        Use :meth:`MergeExporter.export()` instead, which returns a typed
        :class:`~labeling_sft.interfaces.contracts.ExportResult`.
    """
    exporter = MergeExporter()
    result = exporter.export(
        adapter_dir=adapter_dir,
        output_dir=output_dir,
        base_model_id=base_model_id,
    )
    return result.output_path


def convert_to_gguf(model_dir: str, gguf_output: str | None = None) -> str:
    """Convert a merged Hugging Face model to GGUF format.

    .. deprecated::
        Use :meth:`GGUFExporter.export()` instead, which returns a typed
        :class:`~labeling_sft.interfaces.contracts.ExportResult`.
    """
    exporter = GGUFExporter()
    result = exporter.export(
        adapter_dir="",  # not needed for direct conversion
        output_dir=model_dir,
        base_model_id=None,
        gguf_output=gguf_output,
    )
    return result.output_path


# CLI entry point
if __name__ == "__main__":
    import argparse
    import logging

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
