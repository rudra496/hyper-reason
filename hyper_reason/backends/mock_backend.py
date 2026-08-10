"""Deterministic in-process sampler — NOT a model.

``is_live`` is False and ``name`` advertises "heuristic". It exists so the engine mechanics
(MCTS, diversity entropy, FlashKV accounting) can be exercised in tests and in the zero-setup
website demo WITHOUT a model, key, or network. Any output it produces is a clearly-labeled
heuristic, never a model generation.

For real reasoning, use GLMBackend / OllamaBackend / TransformersBackend (is_live=True).
"""

from __future__ import annotations

import hashlib
import re
from typing import Sequence

from .base import Sample, count_tokens

# A small bank of plausible "reasoning step" openers. Deterministically chosen so the same
# prompt always yields the same candidates (reproducible tests / stable website demo).
_STEP_BANK = [
    "Let me identify the given quantities and what's being asked.",
    "First, I'll translate the problem into an arithmetic relationship.",
    "I should compute the primary quantity before applying any reductions.",
    "Now I'll perform the intermediate calculation step by step.",
    "Subtracting the deductions gives the remaining amount.",
    "Let me double-check the arithmetic before committing.",
    "The final answer follows directly from the chain above.",
]


def _seed(prompt: str, i: int) -> int:
    h = hashlib.sha256(f"{prompt}::{i}".encode()).hexdigest()
    return int(h[:8], 16)


class MockBackend:
    name = "mock-deterministic-heuristic"
    is_live = False

    def __init__(self, seed: int = 0):
        self.seed = seed

    def sample(
        self,
        prompt: str,
        k: int = 1,
        temperature: float = 0.7,
        max_tokens: int = 256,
        stop: Sequence[str] = (),
    ) -> list[Sample]:
        nums = re.findall(r"-?\d+(?:\.\d+)?", prompt)
        out: list[Sample] = []
        seen_norm: set[str] = set()
        i = 0
        # Produce up to k distinct (normalized) candidates so diversity entropy is meaningful.
        while len(out) < k and i < k * 4:
            s = _seed(prompt + str(self.seed), i)
            opener = _STEP_BANK[s % len(_STEP_BANK)]
            if nums:
                # Deterministically vary which numbers appear, to create real bucket diversity.
                a = nums[s % len(nums)]
                b = nums[(s + 1) % len(nums)]
                try:
                    prod = float(a) * float(b)
                    prod_str = f"{prod:g}"
                except ValueError:
                    prod_str = f"{a}*{b}"
                text = f"{opener} Using {a} and {b}: intermediate = {prod_str}."
            else:
                text = opener
            text = text[:max_tokens]
            norm = text.lower().strip()
            if norm in seen_norm:
                i += 1
                continue
            seen_norm.add(norm)
            out.append(
                Sample(
                    text=text,
                    prompt_tokens=count_tokens(prompt),
                    completion_tokens=count_tokens(text),
                    finish_reason="stop",
                    latency_ms=0.0,
                    raw={"mock": True},
                )
            )
            i += 1
        # Pad if we couldn't get k distinct (rare) — duplicates are honest about low diversity.
        while len(out) < k:
            out.append(out[-1] if out else Sample(text=".", raw={"mock": True}))
        return out

    def count_tokens(self, text: str) -> int:
        return count_tokens(text)
