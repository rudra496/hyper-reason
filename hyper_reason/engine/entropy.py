"""Honest entropy + priors from real model samples.

We have NO logprobs from the Z.AI gateway, so true next-token policy entropy is unavailable.
Instead we use **sample-diversity entropy**: Shannon entropy over the bucket counts of
*normalized* candidate continuations. This is a sample-based proxy for policy uncertainty,
labeled everywhere as such (never "token entropy").

- ``sample_diversity_entropy``: per-node exploration signal (high = the model is unsure here).
- ``priors_from_diversity``: honest PUCT priors derived from how often each distinct
  continuation was sampled (frequent == higher prior). Never hardcoded.
"""

from __future__ import annotations

from .math_utils import normalize_step, shannon_from_counts


def sample_diversity_entropy(texts: list[str]) -> float:
    """Shannon entropy over normalized-sample buckets. 0 = all samples identical."""
    if not texts:
        return 0.0
    buckets: dict[str, int] = {}
    for t in texts:
        key = normalize_step(t)
        buckets[key] = buckets.get(key, 0) + 1
    return shannon_from_counts(list(buckets.values()))


def priors_from_diversity(texts: list[str]) -> list[float]:
    """Honest per-sample prior = its bucket's share of the total samples."""
    if not texts:
        return []
    normed = [normalize_step(t) for t in texts]
    counts: dict[str, int] = {}
    for n in normed:
        counts[n] = counts.get(n, 0) + 1
    total = len(normed)
    return [counts[n] / total for n in normed]


def bucket_distribution(texts: list[str]) -> dict[str, int]:
    """Raw normalized-bucket counts (for tracing/disclosure)."""
    out: dict[str, int] = {}
    for t in texts:
        n = normalize_step(t)
        out[n] = out.get(n, 0) + 1
    return out
