"""Verifiers — honest answer extraction + self-consistency.

No word-matching "reward" heuristics (those were the v1.x dishonesty). The only value signals
used by the search are: (1) did a branch reach a parseable answer, and (2) do branches agree
(self-consistency). An optional LLM-judge (a DIFFERENT model) can sanity-check format, but
self-consistency is primary and the default.
"""

from __future__ import annotations

from typing import Any

from .math_utils import extract_final_answer


def self_consistency(traces: list[str]) -> tuple[str, float, dict[str, int]]:
    """Majority vote over extracted answers from terminal trajectories.

    Returns (best_answer, confidence, distribution). Unparsable trajectories get their own
    explicit ``__unparsable__`` bucket — never silently counted as a disagreement.
    """
    if not traces:
        return "__unparsable__", 0.0, {}
    answers = [extract_final_answer(t) for t in traces]
    counts: dict[str, int] = {}
    for a in answers:
        counts[a] = counts.get(a, 0) + 1
    best = max(counts, key=counts.get)
    confidence = counts[best] / len(answers)
    return best, round(confidence, 4), counts


class LLMJudge:
    """Optional format/sanity judge. MUST use a different model than the proposer.

    Restricted to well-formedness (not correctness — without ground truth we cannot judge
    correctness, and a same-model judge self-prefers). Off by default; enable via config.judge_model.
    """

    def __init__(self, backend: Any):
        self.backend = backend

    def is_well_formed(self, trajectory: str) -> bool:
        ans = extract_final_answer(trajectory)
        return ans != "__unparsable__"
