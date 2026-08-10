"""OpenAI-compatible chat completions backend (OpenAI, DeepSeek, Groq, vLLM, LMStudio, etc.).

Real HTTP via ``requests``. Token counts come from usage or estimated via count_tokens.
Raises on any failure.
"""

from __future__ import annotations

import os
import time
from typing import Sequence

import requests

from .base import Sample, count_tokens


class OpenAIBackend:
    is_live = True

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 60.0,
        system: str | None = None,
        max_retries: int = 3,
    ):
        self.model = model
        self.base_url = (
            base_url
            or os.environ.get("OPENAI_BASE_URL")
            or os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
        ).rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.timeout = timeout
        self.system = system
        self.max_retries = max_retries
        if not self.api_key and "127.0.0.1" not in self.base_url and "localhost" not in self.base_url:
            raise RuntimeError(
                "OpenAIBackend needs an API key. Set OPENAI_API_KEY (or pass api_key=)."
            )

    @property
    def name(self) -> str:
        return f"openai:{self.model}"

    def _body(self, prompt: str, temperature: float, max_tokens: int, stop: Sequence[str]) -> dict:
        messages = []
        if self.system:
            messages.append({"role": "system", "content": self.system})
        messages.append({"role": "user", "content": prompt})

        body = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages,
        }
        if stop:
            body["stop"] = list(stop)
        return body

    def sample(
        self,
        prompt: str,
        k: int = 1,
        temperature: float = 0.7,
        max_tokens: int = 128,
        stop: Sequence[str] = (),
    ) -> list[Sample]:
        url = f"{self.base_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        body = self._body(prompt, temperature, max_tokens, stop)
        body["n"] = k

        last_err = None
        for attempt in range(self.max_retries):
            try:
                r = requests.post(url, json=body, headers=headers, timeout=self.timeout)
                if r.status_code != 200:
                    raise RuntimeError(f"OpenAI API returned status {r.status_code}: {r.text[:300]}")
                data = r.json()
                choices = data.get("choices", [])
                usage = data.get("usage", {})
                total_ptoks = usage.get("prompt_tokens") or count_tokens(prompt)
                total_ctoks = usage.get("completion_tokens") or 0

                samples = []
                ptok_each = max(1, total_ptoks // max(1, len(choices)))
                ctok_each = max(1, total_ctoks // max(1, len(choices)))

                for c in choices:
                    text = c.get("message", {}).get("content", "") or ""
                    samples.append(
                        Sample(
                            text=text,
                            prompt_tokens=ptok_each,
                            completion_tokens=ctok_each if ctok_each > 0 else count_tokens(text),
                            logprob=None,
                            raw=c,
                        )
                    )
                if len(samples) < k:
                    # Pad if n was ignored by gateway
                    while len(samples) < k and samples:
                        samples.append(samples[0])
                return samples[:k]
            except Exception as e:
                last_err = e
                time.sleep(0.5 * (attempt + 1))
        raise RuntimeError(f"OpenAI API call failed after {self.max_retries} attempts: {last_err}")
