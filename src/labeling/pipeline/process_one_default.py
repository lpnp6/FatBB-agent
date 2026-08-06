"""Interactively extract structured recipe data with a local Ollama model."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from labeling.clients import OllamaLabelingClient
from labeling.prompts import RecipeLabelingPromptBuilder, RecipeRepairPromptBuilder
from labeling.utils.validator import OutputValidationError, OutputValidator


async def process_document(
    document: str,
    client: OllamaLabelingClient,
    validator: OutputValidator | None = None,
) -> dict[str, Any]:
    """Label one document, repairing one invalid model response if needed."""
    validator = validator or OutputValidator()
    result = await client.label(document)
    try:
        return validator.parse(result.raw_output).normalized_json
    except OutputValidationError as error:
        print(f"Validator error: {error}", file=sys.stderr)
        print("Repair: requesting corrected JSON from Ollama...", file=sys.stderr)
        repaired = await client.repair(result.raw_output, str(error))
        print(f"Repair response:\n{repaired.raw_output}", file=sys.stderr)
        return validator.parse(repaired.raw_output).normalized_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.environ.get("OLLAMA_MODEL"), required="OLLAMA_MODEL" not in os.environ)
    parser.add_argument("--host", default=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"))
    parser.add_argument("--file", type=Path, help="Process one Markdown document and exit")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    client = OllamaLabelingClient(
        model=args.model,
        host=args.host,
        label_prompt_builder=RecipeLabelingPromptBuilder(),
        repair_prompt_builder=RecipeRepairPromptBuilder(),
    )

    while True:
        if args.file:
            path = args.file
        else:
            raw_path = input("Document path (empty to quit): ").strip()
            if not raw_path:
                return
            path = Path(raw_path)
        try:
            document = path.expanduser().read_text(encoding="utf-8")
            print(json.dumps(await process_document(document, client), ensure_ascii=False, indent=2))
        except (OSError, OutputValidationError, RuntimeError, ValueError) as error:
            print(f"Error: {error}")
        if args.file:
            return


if __name__ == "__main__":
    asyncio.run(main())
