"""MAAT configuration — model providers, defaults, and settings."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env", override=True)

# Temperature conventions
TEMP_TARGET = 0.3      # Deterministic — target/generator
TEMP_ATTACKER = 0.7    # Creative — attacker/finder (empirically optimal; 0.9 produces more false positives)
TEMP_JUDGE = 0.1       # Consistent — judge/referee

# LangGraph
RECURSION_LIMIT = 200  # Not the default 50

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
LOG_DIR = PROJECT_ROOT / "logs"
RESULTS_DIR = PROJECT_ROOT / "results"
LOG_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


class Provider(str, Enum):
    OLLAMA = "ollama"
    GOOGLE = "google"
    ANTHROPIC = "anthropic"
    OPENAI = "openai"


@dataclass
class ModelConfig:
    """Configuration for a single model."""
    provider: Provider
    model_name: str
    temperature: float = TEMP_TARGET
    max_tokens: int = 4096
    thinking_level: str | None = None  # Gemini 3+ only: 'low', 'medium', 'high', 'minimal'
    request_timeout: float = 180.0     # Per-call timeout in seconds. Prevents indefinite stalls.

    @property
    def display_name(self) -> str:
        return f"{self.provider.value}/{self.model_name}"


# Pre-built model configs for convenience
MODELS = {
    # Local
    "llama3.2:3b": ModelConfig(Provider.OLLAMA, "llama3.2:3b"),
    "mistral": ModelConfig(Provider.OLLAMA, "mistral"),
    "nous-hermes2": ModelConfig(Provider.OLLAMA, "nous-hermes2"),
    # Gemini — Current gen (3.x)
    "gemini-3.1-pro": ModelConfig(Provider.GOOGLE, "gemini-3.1-pro-preview", thinking_level="high"),
    "gemini-3-flash": ModelConfig(Provider.GOOGLE, "gemini-3-flash-preview", thinking_level="high"),
    "gemini-3.1-flash-lite": ModelConfig(Provider.GOOGLE, "gemini-3.1-flash-lite-preview", thinking_level="minimal"),
    # Gemini — Previous gen (2.5)
    "gemini-2.5-flash": ModelConfig(Provider.GOOGLE, "gemini-2.5-flash"),
    "gemini-2.5-pro": ModelConfig(Provider.GOOGLE, "gemini-2.5-pro"),
    # Anthropic
    "haiku": ModelConfig(Provider.ANTHROPIC, "claude-haiku-4-5-20251001"),
    "sonnet": ModelConfig(Provider.ANTHROPIC, "claude-sonnet-4-6"),
    "opus": ModelConfig(Provider.ANTHROPIC, "claude-opus-4-6"),
    # OpenAI
    "gpt-4o": ModelConfig(Provider.OPENAI, "gpt-4o"),
    "o4-mini": ModelConfig(Provider.OPENAI, "o4-mini"),
}
