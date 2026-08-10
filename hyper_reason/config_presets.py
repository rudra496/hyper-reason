"""Preset search budgets for HyperReason.

These are honest BUDGET presets (how much search to buy), not magic "compression" profiles.
Bigger budgets = more model calls = more cost/latency. Disclosed via SearchConfig.disclosure().
"""

from .engine.config import SearchConfig


class SearchPresets:
    @staticmethod
    def ultra_fast() -> SearchConfig:
        """Cheapest search for quick demos / latency-sensitive APIs."""
        return SearchConfig(num_simulations=6, max_depth=3, k_samples=2, temperature=0.5)

    @staticmethod
    def balanced() -> SearchConfig:
        """Default trade-off of cost vs. coverage."""
        return SearchConfig(num_simulations=12, max_depth=4, k_samples=3, temperature=0.7)

    @staticmethod
    def high_accuracy() -> SearchConfig:
        """Deeper, wider search for harder problems (more model calls)."""
        return SearchConfig(num_simulations=24, max_depth=5, k_samples=4, temperature=0.8)
