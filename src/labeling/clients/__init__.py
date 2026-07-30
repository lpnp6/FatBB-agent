"""Concrete model backends for the labeling pipeline."""

from .openai_client import OpenAILabelingClient

__all__ = ["OpenAILabelingClient"]
