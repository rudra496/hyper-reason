"""Ollama backend — salvaged from the original ``ollama_adapter.py``.

SALVAGED — honest because: the HTTP request construction (POST /api/generate with
model/prompt/stream/options) is real and was always real. What changed: the original's
``except URLError`` returned a FAKE "[Local Ollama offline: Fallback simulation step]"
string with invented ``eval_count=35`` (ollama_adapter.py:74-81). That silent fake fallback
is DELETED — this backend now RAISES ConnectionError when the daemon is unreachable, so a
down server can never masquerade as a working model.
"""

from __future__ import annotations

import time
from typing import Sequence

import requests

from .base import Sample, count_tokens


class OllamaBackend:
    is_live = True

    def __init__(
        self,
        model: str = "llama3.1:8b",
        base_url: str = "http://localhost:11434",
        timeout: float = 120.0,
        system: str | None = None,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.system = system

    @property
    def name(self) -> str:
        return f"ollama:{self.model}"

    def is_available(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=3)
            return r.status_code == 200
        except requests.RequestException:
            return False

    def sample(
        self,
        prompt: str,
        k: int = 1,
        temperature: float = 0.7,
        max_tokens: int = 256,
        stop: Sequence[str] = (),
    ) -> list[Sample]:
        url = f"{self.base_url}/api/generate"
        payload: dict = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if self.system:
            payload["system"] = self.system
        if stop:
            payload["stop"] = list(stop)

        out: list[Sample] = []
        for _ in range(k):
            t0 = time.time()
            try:
                resp = requests.post(url, json=payload, timeout=self.timeout)
            except requests.RequestException as e:
                raise ConnectionError(
                    f"Ollama unreachable at {self.base_url}: {e}. "
                    "Start the daemon (`ollama serve`) or choose another backend."
                ) from e
            latency_ms = (time.time() - t0) * 1000.0
            if resp.status_code != 200:
                raise RuntimeError(
                    f"OllamaBackend {self.model} HTTP {resp.status_code}: {resp.text[:300]}"
                )
            data = resp.json()
            out.append(
                Sample(
                    text=data.get("response", ""),
                    prompt_tokens=int(data.get("prompt_eval_count", 0)),
                    completion_tokens=int(data.get("eval_count", 0)),
                    finish_reason="stop" if data.get("done") else "length",
                    latency_ms=latency_ms,
                    raw={"model": self.model},
                )
            )
        return out

    def count_tokens(self, text: str) -> int:
        return count_tokens(text)
