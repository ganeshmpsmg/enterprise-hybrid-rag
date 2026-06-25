"""
OpenAI Connector - Production OpenAI API integration with retry and streaming.
"""
import logging
import os
import time
from typing import Optional

logger = logging.getLogger(__name__)
class OpenAIConnector:
    """
    Production OpenAI API connector.
    """

    COST_PER_1K = {
        "gpt-4o": {"input": 0.005, "output": 0.015},
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        max_tokens: int = 2048,
        temperature: float = 0.1,
        max_retries: int = 3,
        timeout: int = 60,
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        self.timeout = timeout

        self._client = None
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_requests = 0

    @property
    def client(self):
        if self._client is None:
            try:
                from openai import OpenAI

                self._client = OpenAI(
                    api_key=self.api_key,
                    timeout=self.timeout,
                    max_retries=self.max_retries,
                )

            except ImportError:
                raise ImportError(
                    "openai not installed. Run: pip install openai"
                )

        return self._client

    def generate(
        self,
        messages: list[dict],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stream: bool = False,
    ) -> str:

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens or self.max_tokens,
                    temperature=(
                        temperature
                        if temperature is not None
                        else self.temperature
                    ),
                    stream=False,
                )

                usage = response.usage

                self._total_input_tokens += usage.prompt_tokens
                self._total_output_tokens += usage.completion_tokens
                self._total_requests += 1

                content = response.choices[0].message.content

                logger.debug(
                    f"OpenAI generation: "
                    f"{usage.prompt_tokens} input, "
                    f"{usage.completion_tokens} output tokens"
                )

                return content or ""

            except Exception as e:

                print("\n========== OPENAI ERROR ==========")
                print("TYPE:", type(e).__name__)
                print("ERROR:", str(e))
                print("==================================\n")

                error_name = type(e).__name__

                if (
                    "RateLimitError" in error_name
                    and attempt < self.max_retries - 1
                ):
                    wait = 2 ** attempt
                    logger.warning(
                        f"Rate limit hit. Retrying in {wait}s..."
                    )
                    time.sleep(wait)

                elif "AuthenticationError" in error_name:
                    raise ValueError(
                        "Invalid OpenAI API key"
                    ) from e

                elif attempt == self.max_retries - 1:
                    raise RuntimeError(
                        f"OpenAI failed after "
                        f"{self.max_retries} attempts: {e}"
                    ) from e

                else:
                    wait = 2 ** attempt
                    logger.warning(
                        f"OpenAI error ({e}). "
                        f"Retry {attempt+1}/{self.max_retries} "
                        f"in {wait}s"
                    )
                    time.sleep(wait)

        return ""

    def get_stats(self) -> dict:

        costs = self.COST_PER_1K.get(
            self.model,
            {"input": 0, "output": 0},
        )

        estimated_cost = (
            (self._total_input_tokens / 1000) * costs["input"]
            + (self._total_output_tokens / 1000) * costs["output"]
        )

        return {
            "model": self.model,
            "total_requests": self._total_requests,
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "estimated_cost_usd": round(
                estimated_cost,
                6,
            ),
        }