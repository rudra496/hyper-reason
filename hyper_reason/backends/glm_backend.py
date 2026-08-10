"""Z.AI GLM backend (Anthropic-compatible /v1/messages) — the live-tested default.

Real HTTP via ``requests``. Token counts come from the gateway's own ``usage`` report.
On any failure it RAISES (never fabricates a response). Works against any Anthropic-compatible
endpoint (Z.AI, the real Anthropic API, etc.) via base_url/api_key.
"""

from __future__ import annotations

import os
import time
from typing import Sequence

import requests

from .base import Sample, count_tokens


class GLMBackend:
    is_live = True

    def __init__(
        self,
        model: str = "glm-4.6",
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 60.0,
        system: str | None = None,
    ):
        self.model = model
        self.base_url = (
            base_url or os.environ.get("ANTHROPIC_BASE_URL", "https://api.z.ai/api/anthropic")
        ).rstrip("/")
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ZAI_API_KEY")
        self.timeout = timeout
        self.system = system
        if not self.api_key:
            raise RuntimeError(
                "GLMBackend needs an API key. Set ANTHROPIC_API_KEY (or pass api_key=)."
            )

    @property
    def name(self) -> str:
        return f"glm:{self.model}"

    def _body(self, prompt: str, temperature: float, max_tokens: int, stop: Sequence[str]) -> dict:
        body = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if self.system:
            body["system"] = self.system
        if stop:
            body["stop_sequences"] = list(stop)
        return body

    def sample(
        self,
        prompt: str,
        k: int = 1,
        temperature: float = 0.7,
        max_tokens: int = 256,
        stop: Sequence[str] = (),
    ) -> list[Sample]:
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        url = f"{self.base_url}/v1/messages"
        out: list[Sample] = []
        for _ in range(k):
            t0 = time.time()
            resp = requests.post(
                url, headers=headers, json=self._body(prompt, temperature, max_tokens, stop),
                timeout=self.timeout,
            )
            latency_ms = (time.time() - t0) * 1000.0
            if resp.status_code != 200:
                raise RuntimeError(
                    f"GLMBackend {self.model} HTTP {resp.status_code}: {resp.text[:300]}"
                )
            data = resp.json()
            content = data.get("content") or []
            text = content[0].get("text", "") if content else ""
            usage = data.get("usage", {}) or {}
            out.append(
                Sample(
                    text=text,
                    prompt_tokens=int(usage.get("input_tokens", 0)),
                    completion_tokens=int(usage.get("output_tokens", 0)),
                    finish_reason=data.get("stop_reason", "stop") or "stop",
                    latency_ms=latency_ms,
                    raw={"model": self.model, "id": data.get("id")},
                )
            )
        return out

    def count_tokens(self, text: str) -> int:
        return count_tokens(text)
