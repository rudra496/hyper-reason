"""The honest HyperReason engine core.

Pure Python (stdlib + numpy at most). No torch/requests/httpx here — model access goes
through ``hyper_reason.backends``. This keeps the core portable (and JS-portable for the
website) and makes every number traceable to a real backend call.
"""

from .config import SearchConfig
from .math_utils import (
    shannon,
    shannon_from_counts,
    compute_salience,
    extract_boxed,
    extract_final_answer,
    normalize_step,
)
from .entropy import sample_diversity_entropy, priors_from_diversity, bucket_distribution
from .verifier import self_consistency, LLMJudge
from .mcts import ReasonEngine, TreeNode
from .flashkv import FlashKVSimulator

__all__ = [
    "SearchConfig",
    "ReasonEngine",
    "TreeNode",
    "shannon",
    "shannon_from_counts",
    "compute_salience",
    "extract_boxed",
    "extract_final_answer",
    "normalize_step",
    "sample_diversity_entropy",
    "priors_from_diversity",
    "bucket_distribution",
    "self_consistency",
    "LLMJudge",
    "FlashKVSimulator",
]
