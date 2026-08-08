"""
Dynamic KV-Cache Sparsification Module (T-KVP Algorithm)
Novel Scientific Contribution: Entropy-guided attention score decay for KV-cache pruning during test-time tree search.
"""

import math
import time
from typing import Dict, List, Tuple, Optional, Any

class KVCompressionConfig:
    def __init__(
        self,
        pruning_threshold: float = 0.15,
        max_cache_len: int = 4096,
        protected_window_size: int = 64,
        entropy_decay_gamma: float = 0.95,
        target_compression_ratio: float = 0.50
    ):
        self.pruning_threshold = pruning_threshold
        self.max_cache_len = max_cache_len
        self.protected_window_size = protected_window_size
        self.entropy_decay_gamma = entropy_decay_gamma
        self.target_compression_ratio = target_compression_ratio


class DynamicKVCacheCompressor:
    """
    Token-Attentive Dynamic KV-Cache Pruner (T-KVP).
    Maintains a high-density KV-cache during tree search rollouts by dynamically evaluating 
    token salience and pruning non-essential intermediate reasoning steps.
    """
    def __init__(self, config: Optional[KVCompressionConfig] = None):
        self.config = config or KVCompressionConfig()
        self.total_tokens_processed = 0
        self.total_tokens_pruned = 0
        self.compression_history: List[Dict[str, Any]] = []

    def calculate_token_entropy(self, logit_probabilities: List[float]) -> float:
        """Computes Shannon entropy for token probability distributions."""
        entropy = 0.0
        for p in logit_probabilities:
            if p > 1e-9:
                entropy -= p * math.log2(p)
        return entropy

    def compute_salience_scores(
        self, 
        attention_matrix: List[List[float]], 
        token_entropies: List[float]
    ) -> List[float]:
        """
        Computes salience scores by combining global attention weights with local token entropy decay.
        S_i = (sum_j Attn[j, i]) * (1.0 + gamma * Entropy_i)
        """
        num_tokens = len(attention_matrix)
        if num_tokens == 0:
            return []

        salience = [0.0] * num_tokens
        for j in range(num_tokens):
            row = attention_matrix[j]
            for i in range(min(len(row), num_tokens)):
                salience[i] += row[i]

        for i in range(num_tokens):
            entropy_boost = token_entropies[i] if i < len(token_entropies) else 1.0
            salience[i] = salience[i] * (1.0 + self.config.entropy_decay_gamma * entropy_boost)

        return salience

    def prune_cache(
        self, 
        kv_cache_state: Dict[str, Any], 
        attention_weights: List[List[float]],
        token_entropies: List[float]
    ) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """
        Prunes non-critical tokens from KV-cache state while protecting initial prompt context 
        and the sliding recent window.
        """
        start_time = time.time()
        num_tokens = len(attention_weights)
        self.total_tokens_processed += num_tokens

        if num_tokens <= self.config.protected_window_size:
            return kv_cache_state, {
                "pruned_tokens": 0,
                "compression_ratio": 0.0,
                "latency_ms": (time.time() - start_time) * 1000
            }

        salience_scores = self.compute_salience_scores(attention_weights, token_entropies)
        
        # Protect initial system tokens (first 16 tokens) and recent sliding window
        protected_prefix = 16
        protected_suffix = self.config.protected_window_size
        
        indices_to_keep = set(range(min(protected_prefix, num_tokens)))
        indices_to_keep.update(range(max(0, num_tokens - protected_suffix), num_tokens))
        
        candidates = [
            (i, salience_scores[i]) 
            for i in range(num_tokens) 
            if i not in indices_to_keep
        ]
        
        # Sort candidates by salience score descending
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Retain top K tokens based on target compression ratio
        target_retained = int(len(candidates) * (1.0 - self.config.target_compression_ratio))
        retained_candidates = candidates[:target_retained]
        
        for idx, _ in retained_candidates:
            indices_to_keep.add(idx)

        pruned_count = num_tokens - len(indices_to_keep)
        self.total_tokens_pruned += pruned_count

        pruned_cache = {
            "retained_indices": sorted(list(indices_to_keep)),
            "original_length": num_tokens,
            "compressed_length": len(indices_to_keep),
            "memory_saved_mb": (pruned_count * 0.002) # approx size per token
        }

        stats = {
            "pruned_tokens": pruned_count,
            "compression_ratio": pruned_count / max(1, num_tokens),
            "latency_ms": (time.time() - start_time) * 1000,
            "retained_count": len(indices_to_keep)
        }
        
        self.compression_history.append(stats)
        return pruned_cache, stats

    def get_summary(self) -> Dict[str, Any]:
        """Returns aggregate compression efficiency statistics."""
        overall_ratio = (
            self.total_tokens_pruned / max(1, self.total_tokens_processed)
        )
        return {
            "total_processed": self.total_tokens_processed,
            "total_pruned": self.total_tokens_pruned,
            "overall_compression_ratio": round(overall_ratio * 100, 2),
            "estimated_vram_saved_gb": round(self.total_tokens_pruned * 0.000002, 4)
        }
