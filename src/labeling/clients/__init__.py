"""Concrete model backends for the labeling pipeline."""

from .ollama_client import OllamaLabelingClient
from .openai_client import OpenAILabelingClient

__all__ = ["OllamaLabelingClient", "OpenAILabelingClient"]
