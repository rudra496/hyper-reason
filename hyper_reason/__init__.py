"""
HyperReason: Autonomous Test-Time Compute Scaling & Dynamic KV-Cache Sparsification
Author: Rudra Sarker & Buggz
License: MIT
"""

from .mcts_engine import ReasonEngine, TreeNode, SearchConfig
from .kv_compressor import DynamicKVCacheCompressor, KVCompressionConfig
from .verifier import SelfConsistencyVerifier, StepValueEvaluator
from .terminal_visualizer import TreeVisualizer
from .pytorch_wrapper import PyTorchKVCacheHook
from .ollama_adapter import OllamaModelAdapter
from .vllm_adapter import VLLMPagedAttentionHook
from .datasets import GSM8KDataset, BenchmarkEvaluator
from .flash_kv import FlashKVTreeManager, KVBlock
from .speculative import SpeculativeTreeEngine
from .wrapper import wrap_model, WrappedReasoningModel
from .exporters import TreeTraceExporter
from .config_presets import SearchPresets
from .cost_analyzer import CostEfficiencyAnalyzer
from .model_quantizer import KVQuantizer
from .multi_agent_tree import MultiAgentReasonTree
from .tree_visualizer_gui import HTMLTreeVisualizer

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
    "VLLMPagedAttentionHook",
    "GSM8KDataset",
    "BenchmarkEvaluator",
    "FlashKVTreeManager",
    "KVBlock",
    "SpeculativeTreeEngine",
    "wrap_model",
    "WrappedReasoningModel",
    "TreeTraceExporter",
    "SearchPresets",
    "CostEfficiencyAnalyzer",
    "KVQuantizer",
    "MultiAgentReasonTree",
    "HTMLTreeVisualizer",
]
