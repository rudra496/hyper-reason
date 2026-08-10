"""SearchConfig — the locked search configuration (Phase 0 contract).

Fields are deliberately explicit and disclosed, because the headline entropy/accuracy
numbers are functions of this config (temperature especially). Anything that prints a
metric must also print the config that produced it.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SearchConfig:
    # Budget (stateless gateway -> no cross-sibling KV reuse, so these bound cost)
    num_simulations: int = 32   # hard cap on total node expansions
    max_depth: int = 4          # max reasoning-chain depth
    k_samples: int = 3          # candidate next-steps sampled per expansion

    # Sampling (DISCLOSED — entropy is a function of temperature)
    temperature: float = 0.7
    top_p: float = 1.0
    max_tokens_per_step: int = 128

    # AE-PUCT selection
    c_puct: float = 1.414
    entropy_alpha: float = 0.15  # weight on sample_diversity_entropy in AE-PUCT

    # Verifier
    judge_model: str | None = None  # if set, LLM-judge; MUST differ from proposer model

    # Termination
    answer_format_hint: str = r"Put the final answer in \boxed{}."

    def disclosure(self) -> dict:
        """Everything that must accompany a reported metric."""
        return {
            "num_simulations": self.num_simulations,
            "max_depth": self.max_depth,
            "k_samples": self.k_samples,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens_per_step": self.max_tokens_per_step,
            "c_puct": self.c_puct,
            "entropy_alpha": self.entropy_alpha,
            "entropy_source": "sample_diversity_entropy (K samples; no logprobs via Z.AI gateway)",
            "judge_model": self.judge_model,
        }
