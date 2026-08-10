"""Shared pure functions for the engine.

Two functions (``shannon``, ``compute_salience``) are salvaged from the original
``kv_compressor.py`` because their *math* is correct; the dishonest byte-constants that
accompanied them (``* 0.002`` MB etc.) were dropped.

SALVAGED — honest because: these are textbook formulas (Shannon entropy; column-sum
attention salience boosted by entropy) with no fabricated constants. Inputs are supplied
by the caller from real measurements; outputs are labeled by the caller.
"""

from __future__ import annotations

import math
import re
from typing import Sequence


def shannon(probs: Sequence[float]) -> float:
    """Shannon entropy (base-2) of a probability distribution.

    SALVAGED from kv_compressor.calculate_token_entropy. H = -sum p*log2(p).
    """
    h = 0.0
    for p in probs:
        if p > 1e-12:
            h -= p * math.log2(p)
    return h


def shannon_from_counts(counts: Sequence[int]) -> float:
    """Shannon entropy over a frequency histogram (normalizes counts to probs).

    Used for sample-diversity entropy: pass the bucket sizes of normalized candidate
    next-steps. This is the honest, logprob-free node-exploration signal.
    """
    total = sum(counts)
    if total <= 0:
        return 0.0
    return shannon([c / total for c in counts if c > 0])


def compute_salience(
    attention_matrix: Sequence[Sequence[float]],
    token_entropies: Sequence[float],
    gamma: float = 0.95,
) -> list[float]:
    """Per-token salience = column attention mass * (1 + gamma * entropy).

    SALVAGED from kv_compressor.compute_salience_scores. S_i = (sum_j Attn[j,i]) * (1 + gamma*H_i).
    No fabricated constants; ``gamma`` is an explicit, disclosed parameter.
    """
    n = len(attention_matrix)
    if n == 0:
        return []
    col_mass = [0.0] * n
    for j in range(n):
        row = attention_matrix[j]
        for i in range(min(len(row), n)):
            col_mass[i] += row[i]
    return [
        col_mass[i] * (1.0 + gamma * (token_entropies[i] if i < len(token_entropies) else 0.0))
        for i in range(n)
    ]


_BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def extract_boxed(text: str) -> str | None:
    """Return the last ``\\boxed{...}`` payload, stripped; None if absent."""
    matches = _BOXED_RE.findall(text)
    if not matches:
        return None
    ans = matches[-1].strip()
    return ans if ans else None


def normalize_step(text: str) -> str:
    """Normalize a candidate step for diversity bucketing.

    Lowercase, strip ``\\boxed{...}`` payloads to a placeholder, collapse whitespace.
    Two semantically-identical steps that differ only by answer formatting/case then bucket
    together — which is what we want for a diversity signal.
    """
    t = _BOXED_RE.sub("[boxed]", text)
    t = t.lower()
    t = re.sub(r"\s+", " ", t).strip()
    return t


def extract_final_answer(text: str) -> str:
    """Best-effort final answer: last \\boxed payload, else last number, else marker."""
    boxed = extract_boxed(text)
    if boxed is not None:
        return boxed
    nums = _NUM_RE.findall(text)
    if nums:
        return nums[-1]
    return "__unparsable__"
