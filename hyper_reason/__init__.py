"""
HyperReason: Autonomous Test-Time Compute Scaling & Dynamic KV-Cache Sparsification
Author: Rudra Sarker (Rudra Sir) & Buggz
License: MIT
"""

from .mcts_engine import ReasonEngine, TreeNode, SearchConfig
from .kv_compressor import DynamicKVCacheCompressor, KVCompressionConfig
from .verifier import SelfConsistencyVerifier, StepValueEvaluator
from .terminal_visualizer import TreeVisualizer
from .pytorch_wrapper import PyTorchKVCacheHook
from .ollama_adapter import OllamaModelAdapter

__version__ = "1.0.0"
__author__ = "Rudra Sarker"

__all__ = [
    "ReasonEngine",
    "TreeNode",
    "SearchConfig",
    "DynamicKVCacheCompressor",
    "KVCompressionConfig",
    "SelfConsistencyVerifier",
    "StepValueEvaluator",
    "TreeVisualizer",
    "PyTorchKVCacheHook",
    "OllamaModelAdapter",
]
