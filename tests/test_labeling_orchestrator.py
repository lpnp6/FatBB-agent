"""End-to-end behavior tests for the bootstrap labeling pipeline."""

from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from labeling.bootstrap.checkpoint import CheckpointManager
from labeling.dedup.simhash_store import SimHashDedupStore
from labeling.interfaces.dedup_store import HashStatus
from labeling.interfaces.labeling_client import LabelingClient
from labeling.models import ExtractionResult
from labeling.bootstrap.orchestrator import JsonlTrainingWriter, LabelingPipeline


RECIPE_OUTPUT = json.dumps({
    "dish": {
        "name": "Lemon Pasta", "aliases": [], "dish_type": "main_dish",
        "taste_profile": ["sour"], "dietary": [], "cooking_time_min": 10,
        "prep_time_min": 5, "total_time_min": 15, "difficulty": "easy",
        "servings": 2, "calories_per_serving": None, "description": None,
        "cooking_steps": [{"order": 1, "method": "boil", "method_name": "Boil", "ingredient_refs": ["pasta"], "note": None, "duration_min": 10, "heat_level": None}],
        "cuisine": None,
    },
    "ingredients": [{"name": "Pasta", "category": "grain", "amount": "200g", "amount_normalized": {"value": 200, "unit": "g", "range_low": None, "range_high": None}, "is_essential": True, "preparation": None, "notes": None}],
    "dish_relations": [], "ingredient_relations": [],
})
NON_RECIPE_OUTPUT = '{"dish":null,"ingredients":[],"ingredient_relations":[],"reason":"not_a_recipe"}'


class FakeClient(LabelingClient):
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = outputs
        self.calls = 0

    @property
    def model_name(self) -> str:
        return "fake"

    async def label(self, markdown: str) -> ExtractionResult:
        output = self.outputs[self.calls]
        self.calls += 1
        return ExtractionResult(raw_output=output, model=self.model_name)

    async def repair(self, raw_output: str, error_message: str) -> ExtractionResult:
        return ExtractionResult(raw_output=raw_output, model=self.model_name)


class LabelingPipelineTests(unittest.TestCase):
    def test_recipe_is_written_then_deduped_and_checkpointed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "recipe.md"
            source.write_text("# Lemon Pasta", encoding="utf-8")
            fingerprint = "0123456789abcdef"
            store = SimHashDedupStore(root / "dedup.db")
            store.register(fingerprint, str(source), HashStatus.IN_FLIGHT)
            client = FakeClient([RECIPE_OUTPUT])
            checkpoint_path = root / "checkpoint.json"
            output_path = root / "training.jsonl"
            pipeline = LabelingPipeline(
                client=client, dedup_store=store,
                checkpoint=CheckpointManager(checkpoint_path, manifest_path=root / "manifest.jsonl", output_path=output_path),
                training_writer=JsonlTrainingWriter(output_path), retries=0,
            )
            manifest = [{"id": "labeling:corpus:recipe.md", "path": str(source), "recipe_card_hash": fingerprint}]

            self.assertEqual(asyncio.run(pipeline.run(manifest)), {"completed": 1})
            self.assertEqual(store.lookup(fingerprint), HashStatus.ACCEPTED)
            record = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(record["output"]["dish"]["name"], "Lemon Pasta")
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            self.assertEqual(checkpoint["items"][manifest[0]["id"]]["output_line"], 1)

            self.assertEqual(asyncio.run(pipeline.run(manifest)), {"skipped_completed": 1})
            self.assertEqual(client.calls, 1)
            store._db.close()

    def test_non_recipe_is_saved_for_training_without_graph_model_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "guide.md"
            source.write_text("Gift guide", encoding="utf-8")
            fingerprint = "fedcba9876543210"
            store = SimHashDedupStore(root / "dedup.db")
            store.register(fingerprint, str(source), HashStatus.IN_FLIGHT)
            output_path = root / "training.jsonl"
            pipeline = LabelingPipeline(
                client=FakeClient([NON_RECIPE_OUTPUT]), dedup_store=store,
                checkpoint=CheckpointManager(root / "checkpoint.json", manifest_path=root / "manifest.jsonl", output_path=output_path),
                training_writer=JsonlTrainingWriter(output_path), retries=0,
            )
            manifest = [{"id": "labeling:corpus:guide.md", "path": str(source), "recipe_card_hash": fingerprint}]

            self.assertEqual(asyncio.run(pipeline.run(manifest)), {"not_a_recipe": 1})
            self.assertEqual(store.lookup(fingerprint), HashStatus.ACCEPTED)
            record = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertTrue(record["is_not_a_recipe"])
            self.assertEqual(record["output"]["reason"], "not_a_recipe")
            store._db.close()
