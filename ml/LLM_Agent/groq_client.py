"""
High-Performance Groq Cloud Client Wrapper.
Provides robust inference execution with automatic retries, fallback routing, and error shielding.
"""

import time
import logging
from typing import List, Dict, Any, Optional, Iterator
from groq import Groq, APIError, RateLimitError, APIConnectionError
from LLM_Agent.config import AgentConfig

logger = logging.getLogger("GroqClient")

class GroqClientWrapper:
    """Wrapper around Groq SDK with resilience, fallbacks, and parameter enforcement."""
    
    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()
        self.config.validate()
        self.client = Groq(api_key=self.config.groq_api_key, timeout=self.config.timeout_seconds)
        self.active_model = self.config.primary_model

    def generate_chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ) -> Any:
        """
        Sends chat completion request to Groq API with automatic retry and model fallback.
        """
        target_model = model or self.active_model
        temp = temperature if temperature is not None else self.config.temperature
        tokens = max_tokens or self.config.max_completion_tokens

        last_error = None
        for attempt in range(1, self.config.max_retries + 1):
            try:
                if stream:
                    return self.client.chat.completions.create(
                        model=target_model,
                        messages=messages,
                        temperature=temp,
                        max_tokens=tokens,
                        top_p=self.config.top_p,
                        stream=True
                    )
                else:
                    response = self.client.chat.completions.create(
                        model=target_model,
                        messages=messages,
                        temperature=temp,
                        max_tokens=tokens,
                        top_p=self.config.top_p,
                        stream=False
                    )
                    return response.choices[0].message.content

            except RateLimitError as rle:
                logger.warning(f"Groq Rate Limit on {target_model} (attempt {attempt}/{self.config.max_retries}): {rle}")
                time.sleep(2 ** attempt)
                last_error = rle
            except APIConnectionError as ace:
                logger.warning(f"Groq Connection Error (attempt {attempt}/{self.config.max_retries}): {ace}")
                time.sleep(1.5 ** attempt)
                last_error = ace
            except APIError as ape:
                logger.error(f"Groq API Error on {target_model}: {ape}")
                # Try fallback model if primary model fails
                if target_model != self.config.fallback_model:
                    logger.info(f"Switching from {target_model} to fallback {self.config.fallback_model}")
                    target_model = self.config.fallback_model
                last_error = ape

        raise RuntimeError(f"Failed to generate Groq completion after {self.config.max_retries} attempts. Last error: {last_error}")

    def test_connection(self) -> Dict[str, Any]:
        """Validates live connectivity to Groq API."""
        start_t = time.perf_counter()
        resp = self.generate_chat_completion(
            messages=[
                {"role": "system", "content": "Respond strictly with: OK"},
                {"role": "user", "content": "Ping"}
            ],
            max_tokens=10
        )
        latency_ms = (time.perf_counter() - start_t) * 1000
        return {
            "status": "connected" if "OK" in resp else "degraded",
            "model": self.active_model,
            "response": resp.strip(),
            "latency_ms": round(latency_ms, 2)
        }
