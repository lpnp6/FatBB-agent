"""Unit tests for the OpenAI-compatible labeling client."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

from labeling.clients.openai_client import OpenAILabelingClient
from labeling.interfaces.prompt_builder import PromptBuilder


class StubPromptBuilder(PromptBuilder):
    def build_messages(self, *args: object, **kwargs: object) -> list[dict[str, str]]:
        return [{"role": "user", "content": str(args[0] if args else "")}]


class FakeCompletions:
    def __init__(self, response: object) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    async def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        return self.response


class OpenAILabelingClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_label_normalizes_a_chat_completion_response(self) -> None:
        response = SimpleNamespace(
            id="chatcmpl_test",
            choices=[SimpleNamespace(
                message=SimpleNamespace(content='{"dish": null, "reason": "not_a_recipe"}'),
                finish_reason="stop",
            )],
            usage=SimpleNamespace(prompt_tokens=12, completion_tokens=7),
        )
        completions = FakeCompletions(response)
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        stub = StubPromptBuilder()
        labeling_client = OpenAILabelingClient(
            api_key="test-key",
            model="test-model",
            label_prompt_builder=stub,
            repair_prompt_builder=stub,
            base_url="https://example.test/v1",
            max_concurrent=2,
            client=client,
        )

        with self.assertLogs("labeling.clients.openai_client", level="INFO") as logs:
            result = await labeling_client.label("not a recipe")

        self.assertEqual(result.raw_output, '{"dish": null, "reason": "not_a_recipe"}')
        self.assertEqual(result.model, "test-model")
        self.assertEqual(result.token_usage.input, 12)
        self.assertEqual(result.token_usage.output, 7)
        self.assertEqual(result.metadata["provider_response_id"], "chatcmpl_test")
        self.assertEqual(result.metadata["finish_reason"], "stop")
        self.assertIn("Dispatching labeling request model=test-model input_chars=12", logs.output[0])
        self.assertIn("Labeling request completed model=test-model", logs.output[1])
        self.assertIn("input_tokens=12 output_tokens=7", logs.output[1])
        self.assertEqual(completions.calls, [{
            "model": "test-model",
            "messages": [{"role": "user", "content": "not a recipe"}],
            "max_tokens": 16384,
            "response_format": {"type": "json_object"},
        }])

    async def test_label_rejects_an_empty_completion(self) -> None:
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=None), finish_reason="stop")],
            usage=None,
        )
        client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions(response)))
        stub = StubPromptBuilder()
        labeling_client = OpenAILabelingClient(
            api_key="test-key", model="test-model",
            label_prompt_builder=stub, repair_prompt_builder=stub,
            client=client, empty_response_retries=0,
        )

        with self.assertRaisesRegex(RuntimeError, "empty completion"):
            await labeling_client.label("recipe")


if __name__ == "__main__":
    unittest.main()
