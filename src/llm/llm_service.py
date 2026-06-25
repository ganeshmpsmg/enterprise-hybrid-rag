"""
LLM Service - Unified interface supporting OpenAI, Anthropic, and Ollama.
Abstracts provider-specific details for clean RAG pipeline integration.
"""
import logging
import os
from typing import Optional

from src.llm.openai_connector import OpenAIConnector

logger = logging.getLogger(__name__)


class LLMService:
    """
    Unified LLM service supporting multiple providers.

    Supported providers:
    - openai: GPT-4o, GPT-4o-mini (default)
    - anthropic: Claude 3 Haiku, Sonnet, Opus
    - ollama: Llama 3, Mistral, Gemma (local)

    Selection via LLM_PROVIDER environment variable or constructor.
    """

    def __init__(
        self,
        provider: str = "openai",
        model: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        **kwargs,
    ):
        self.provider = provider.lower()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._connector = self._init_connector(model, **kwargs)

    def generate(
        self,
        messages: list[dict],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Generate response from the configured LLM.

        Args:
            messages: Conversation messages in OpenAI format
            max_tokens: Token limit for response
            temperature: Sampling temperature (0=deterministic, 1=creative)

        Returns:
            Generated response string
        """
        return self._connector.generate(
            messages=messages,
            max_tokens=max_tokens or self.max_tokens,
            temperature=temperature if temperature is not None else self.temperature,
        )

    def generate_from_prompt(self, system: str, user: str, **kwargs) -> str:
        """Convenience method for simple system+user prompt."""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        return self.generate(messages, **kwargs)

    def get_stats(self) -> dict:
        return {"provider": self.provider, **self._connector.get_stats()}

    def _init_connector(self, model: Optional[str], **kwargs):
        """Initialize the appropriate LLM connector."""
        if self.provider == "openai":
            return OpenAIConnector(
                model=model or "gpt-4o-mini",
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                **kwargs,
            )
        elif self.provider == "anthropic":
            return self._init_anthropic(model, **kwargs)
        elif self.provider == "ollama":
            return self._init_ollama(model, **kwargs)
        else:
            logger.warning(f"Unknown provider '{self.provider}', defaulting to OpenAI")
            return OpenAIConnector(model=model or "gpt-4o-mini")

    def _init_anthropic(self, model: Optional[str], **kwargs):
        """Initialize Anthropic connector."""
        class AnthropicConnector:
            def __init__(self, model, temperature, max_tokens):
                self.model = model or "claude-3-haiku-20240307"
                self.temperature = temperature
                self.max_tokens = max_tokens
                self._total = 0

            def generate(self, messages, max_tokens=None, temperature=None, **kw):
                try:
                    import anthropic
                    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
                    system_msg = next((m["content"] for m in messages if m["role"] == "system"), "")
                    user_msgs = [m for m in messages if m["role"] != "system"]
                    resp = client.messages.create(
                        model=self.model,
                        max_tokens=max_tokens or self.max_tokens,
                        temperature=temperature or self.temperature,
                        system=system_msg,
                        messages=user_msgs,
                    )
                    self._total += 1
                    return resp.content[0].text
                except Exception as e:
                    raise RuntimeError(f"Anthropic error: {e}") from e

            def get_stats(self):
                return {"model": self.model, "total_requests": self._total}

        return AnthropicConnector(model, self.temperature, self.max_tokens)

    def _init_ollama(self, model: Optional[str], **kwargs):
        """Initialize Ollama (local LLM) connector."""
        class OllamaConnector:
            def __init__(self, model, temperature, max_tokens):
                self.model = model or "llama3.2"
                self.temperature = temperature
                self.max_tokens = max_tokens
                self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
                self._total = 0

            def generate(self, messages, max_tokens=None, temperature=None, **kw):
                try:
                    import requests
                    prompt = "\n".join(
                        f"{m['role'].upper()}: {m['content']}" for m in messages
                    )
                    resp = requests.post(
                        f"{self.base_url}/api/generate",
                        json={
                            "model": self.model,
                            "prompt": prompt,
                            "stream": False,
                            "options": {
                                "temperature": temperature or self.temperature,
                                "num_predict": max_tokens or self.max_tokens,
                            },
                        },
                        timeout=120,
                    )
                    resp.raise_for_status()
                    self._total += 1
                    return resp.json().get("response", "")
                except Exception as e:
                    raise RuntimeError(f"Ollama error: {e}") from e

            def get_stats(self):
                return {"model": self.model, "total_requests": self._total}

        return OllamaConnector(model, self.temperature, self.max_tokens)
