"""Backend protocol + the Sample result type.

This is the locked contract (Phase 0). Every backend returns real ``Sample`` objects whose
token counts come from the model's own usage report where one exists. No silent fake
fallbacks: if a backend cannot reach its model it MUST raise, never fabricate plausible text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence, runtime_checkable

try:  # real tokenizer if available -> honest token counts for arbitrary text
    import tiktoken as _tik  # type: ignore

    _ENC = _tik.get_encoding("cl100k_base")
except Exception:  # pragma: no cover - fallback path
    _tik = None
    _ENC = None


@dataclass(frozen=True)
class Sample:
    """One model generation. Token counts are the model's own usage where available."""

    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str = "stop"
    latency_ms: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@runtime_checkable
class ModelBackend(Protocol):
    """Anything that can produce K candidate continuations for a prompt."""

    name: str
    is_live: bool  # True iff this calls a real model (never True for Mock)

    def sample(
        self,
        prompt: str,
        k: int = 1,
        temperature: float = 0.7,
        max_tokens: int = 256,
        stop: Sequence[str] = (),
    ) -> list[Sample]:
        """Return exactly ``k`` real generations. Raise on failure — never fake."""
        ...

    def count_tokens(self, text: str) -> int:
        """Honest token count for ``text``.

        Uses tiktoken (cl100k_base) if installed; otherwise a clearly-labeled heuristic
        (~1.3 tokens/whitespace-token for English). The simulator reports which path was
        used so no consumer mistakes the estimate for the model's own tokenizer.
        """
        ...


def count_tokens(text: str) -> int:
    """Module-level helper backing ``ModelBackend.count_tokens`` default impls."""
    if _ENC is not None:
        return len(_ENC.encode(text))
    # Labeled approximation (no real tokenizer available).
    return max(1, int(len(text.split()) * 1.3))


_TOKENIZER_LIVE = _ENC is not None
