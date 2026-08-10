"""HyperReason — honest test-time compute (AE-MCTS) + projected KV-cache memory accounting.

v2: every claim is backed by code that runs. The engine genuinely drives a model backend
(GLM default / Ollama / Transformers / deterministic Mock). See README for the honesty notes.
"""

from .backends import (
    Sample,
    ModelBackend,
    count_tokens,
    MockBackend,
    GLMBackend,
    OllamaBackend,
    TransformersBackend,
)
from .engine import (
    SearchConfig,
    shannon,
    shannon_from_counts,
    compute_salience,
    extract_boxed,
    extract_final_answer,
    normalize_step,
    sample_diversity_entropy,
    priors_from_diversity,
    self_consistency,
)
from .engine.mcts import ReasonEngine, TreeNode
from .wrapper import wrap_model, WrappedReasoningModel
from .exporters import TreeTraceExporter
from .terminal_visualizer import TreeVisualizer
from .config_presets import SearchPresets
from .agent_memory import ReasoningMemoryStore
from .cost_analyzer import CostEfficiencyAnalyzer

__version__ = "2.0.0.dev0"
__author__ = "Rudra Sarker"

__all__ = [
    # backends
    "Sample", "ModelBackend", "count_tokens",
    "MockBackend", "GLMBackend", "OllamaBackend", "TransformersBackend",
    # engine
    "SearchConfig", "ReasonEngine", "TreeNode",
    "shannon", "shannon_from_counts", "compute_salience",
    "extract_boxed", "extract_final_answer", "normalize_step",
    "sample_diversity_entropy", "priors_from_diversity", "self_consistency",
    # high-level API
    "wrap_model", "WrappedReasoningModel",
    # utilities
    "TreeTraceExporter", "TreeVisualizer", "SearchPresets",
    "ReasoningMemoryStore", "CostEfficiencyAnalyzer",
]
