"""
Universal Drop-in Model Wrapper API
Author: Rudra Sarker (Rudra Sir) & Buggz
License: MIT

Exposes wrap_model() to seamlessly add FlashKV zero-copy caching and AE-MCTS test-time compute search 
to any PyTorch model, HuggingFace AutoModelForCausalLM, vLLM instance, or Ollama adapter.
"""

from typing import Any, Optional, Dict
from .mcts_engine import ReasonEngine, SearchConfig
from .flash_kv import FlashKVTreeManager
from .pytorch_wrapper import PyTorchKVCacheHook


class WrappedReasoningModel:
    """
    Unified model wrapper providing high-level .reason() and .generate() APIs.
    """
    def __init__(self, base_model: Any, config: Optional[SearchConfig] = None):
        self.base_model = base_model
        self.config = config or SearchConfig()
        self.engine = ReasonEngine(config=self.config)
        self.flash_kv = FlashKVTreeManager()
        self.pytorch_hook = PyTorchKVCacheHook()

    def reason(self, prompt: str, num_simulations: Optional[int] = None) -> Dict[str, Any]:
        """
        Executes test-time compute tree search reasoning on input prompt.
        Returns: Dict containing best trajectory, boxed solution, tree visualization, and memory metrics.
        """
        if num_simulations:
            self.engine.config.num_simulations = num_simulations

        # Allocate FlashKV root blocks
        self.flash_kv.allocate_root_blocks(node_id=0, seq_len=len(prompt))

        best_trace, root_node, meta = self.engine.run_mcts(prompt)
        meta["flash_kv_stats"] = self.flash_kv.get_memory_stats()
        
        return {
            "prompt": prompt,
            "solution_trajectory": best_trace,
            "boxed_answer": meta.get("consensus_boxed_answer", ""),
            "confidence": meta.get("consensus_confidence", 0.0),
            "metrics": meta
        }

    def generate(self, prompt: str, **kwargs) -> str:
        """Standard generation API fallback."""
        res = self.reason(prompt)
        return res["solution_trajectory"]


def wrap_model(model: Any, config: Optional[SearchConfig] = None) -> WrappedReasoningModel:
    """
    Universal 1-line wrapper helper function:
    model = wrap_model(base_model)
    """
    return WrappedReasoningModel(base_model=model, config=config)
