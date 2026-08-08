"""
Preset Search Configurations Module
Author: Rudra Sarker (Rudra Sir) & Buggz
License: MIT

Provides hardware-optimized preset search profiles for edge devices, consumer GPUs, and server clusters.
"""

from .mcts_engine import SearchConfig


class SearchPresets:
    """
    Factory presets for SearchConfig tailored to distinct deployment targets.
    """
    @staticmethod
    def ultra_fast() -> SearchConfig:
        """Fastest search mode for edge devices or low-latency APIs."""
        return SearchConfig(
            num_simulations=12,
            max_depth=4,
            top_k_actions=2,
            c_puct=1.2,
            prune_kv_cache=True
        )

    @staticmethod
    def high_accuracy() -> SearchConfig:
        """Deep search tree exploration for complex mathematical / coding benchmarks."""
        return SearchConfig(
            num_simulations=64,
            max_depth=8,
            top_k_actions=4,
            c_puct=1.618,
            prune_kv_cache=True
        )

    @staticmethod
    def speculative_boost() -> SearchConfig:
        """Optimized for speculative parallel draft rollouts."""
        return SearchConfig(
            num_simulations=32,
            max_depth=6,
            top_k_actions=4,
            c_puct=1.414,
            prune_kv_cache=True
        )

    @staticmethod
    def extreme_compression() -> SearchConfig:
        """Maximum KV pruning and FlashKV zero-copy sharing for GPUs with < 8GB VRAM."""
        return SearchConfig(
            num_simulations=24,
            max_depth=5,
            top_k_actions=3,
            c_puct=1.414,
            prune_kv_cache=True
        )
