"""Extract structured recipe data from one file with a local Ollama model."""

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

MAX_REPAIR_ATTEMPTS = 5


async def process_document(
    document: str,
    client: OllamaLabelingClient,
    validator: OutputValidator | None = None,
) -> dict[str, Any]:
    """Label one document, looping repair until valid or attempts exhausted."""
    validator = validator or OutputValidator()
    result = await client.label(document)
    raw = result.raw_output

    for attempt in range(MAX_REPAIR_ATTEMPTS + 1):
        try:
            parsed = validator.parse(raw).normalized_json
            print(f"[attempt {attempt + 1}] ✓ valid", file=sys.stderr)
            return parsed
        except OutputValidationError as error:
            print(f"[attempt {attempt + 1}] ✗ {error}", file=sys.stderr)
            if attempt >= MAX_REPAIR_ATTEMPTS:
                raise
            print(f"[attempt {attempt + 1}] → repairing...", file=sys.stderr)
            result = await client.repair(raw, str(error))
            raw = result.raw_output
            print(f"[attempt {attempt + 1}] repair response:\n{raw}", file=sys.stderr)

    raise RuntimeError(f"Failed after {MAX_REPAIR_ATTEMPTS} repair attempts")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.environ.get("OLLAMA_MODEL"), required="OLLAMA_MODEL" not in os.environ)
    parser.add_argument("--host", default=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"))
    parser.add_argument("--file", required=True, type=Path, help="Markdown document to process")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    client = OllamaLabelingClient(
        model=args.model,
        host=args.host,
        label_prompt_builder=RecipeLabelingPromptBuilder(),
        repair_prompt_builder=RecipeRepairPromptBuilder(),
    )

    try:
        document = args.file.expanduser().read_text(encoding="utf-8")
        print(json.dumps(await process_document(document, client), ensure_ascii=False, indent=2))
    except (OSError, OutputValidationError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
