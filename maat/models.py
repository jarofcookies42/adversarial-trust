"""LLM provider abstraction — unified interface for Ollama, Gemini, Claude, OpenAI."""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel

from maat.config import ModelConfig, Provider


def get_model(config: ModelConfig) -> BaseChatModel:
    """Return a LangChain chat model from a ModelConfig.
    
    This is the single point of abstraction for all LLM providers.
    Every agent in every domain should use this to get its model.
    """
    match config.provider:
        case Provider.OLLAMA:
            from langchain_ollama import ChatOllama
            return ChatOllama(
                model=config.model_name,
                temperature=config.temperature,
                num_predict=config.max_tokens,
            )
        case Provider.GOOGLE:
            from langchain_google_genai import ChatGoogleGenerativeAI
            kwargs = {
                "model": config.model_name,
                "temperature": config.temperature,
                "max_output_tokens": config.max_tokens,
                "request_timeout": config.request_timeout,
            }
            if config.thinking_level:
                kwargs["thinking_level"] = config.thinking_level
            return ChatGoogleGenerativeAI(**kwargs)
        case Provider.ANTHROPIC:
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(
                model=config.model_name,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                timeout=config.request_timeout,
            )
        case Provider.OPENAI:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=config.model_name,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )
        case _:
            raise ValueError(f"Unknown provider: {config.provider}")
