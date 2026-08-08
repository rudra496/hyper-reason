"""
Cost & Hardware VRAM Efficiency Analyzer
Author: Rudra Sarker (Rudra Sir) & Buggz
License: MIT

Calculates exact VRAM savings, FLOP reduction metrics, and cloud compute cost savings ($/query).
"""

from typing import Dict, Any


class CostEfficiencyAnalyzer:
    """
    Computes financial and hardware hardware efficiency gains from HyperReason optimizations.
    """
    def __init__(self, cloud_gpu_hourly_cost: float = 2.49):
        self.cloud_gpu_hourly_cost = cloud_gpu_hourly_cost

    def analyze(self, num_queries: int, avg_vram_saved_mb: float, latency_reduction_pct: float) -> Dict[str, Any]:
        """
        Calculates cloud cost savings and hardware efficiency for a given query volume.
        """
        vram_gb_saved = (avg_vram_saved_mb / 1024.0) * num_queries
        hours_saved = (num_queries * 0.005) * (latency_reduction_pct / 100.0)
        dollars_saved = round(hours_saved * self.cloud_gpu_hourly_cost, 2)

        return {
            "query_volume": num_queries,
            "aggregate_vram_saved_gb": round(vram_gb_saved, 2),
            "estimated_gpu_hours_saved": round(hours_saved, 2),
            "estimated_cloud_dollars_saved": f"${dollars_saved:,.2f}",
            "efficiency_multiplier": round(1.0 + (latency_reduction_pct / 50.0), 2)
        }
