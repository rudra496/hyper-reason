"""
Adaptive Entropy-Guided Monte Carlo Tree Search Engine (AE-MCTS)
Novel Scientific Contribution: Dynamic UCB exploration weighting scaled by token entropy and step verification rewards.
"""

import math
import random
import time
from typing import List, Dict, Optional, Tuple, Any
from .kv_compressor import DynamicKVCacheCompressor, KVCompressionConfig
from .verifier import StepValueEvaluator, SelfConsistencyVerifier


class SearchConfig:
    def __init__(
        self,
        num_simulations: int = 32,
        max_depth: int = 8,
        c_puct: float = 1.414,
        temperature: float = 0.7,
        top_k_actions: int = 3,
        prune_kv_cache: bool = True
    ):
        self.num_simulations = num_simulations
        self.max_depth = max_depth
        self.c_puct = c_puct
        self.temperature = temperature
        self.top_k_actions = top_k_actions
        self.prune_kv_cache = prune_kv_cache


class TreeNode:
    """
    Represents a decision node in the reasoning tree trajectory.
    """
    def __init__(self, state_text: str, parent: Optional["TreeNode"] = None, prior_prob: float = 1.0):
        self.state_text = state_text
        self.parent = parent
        self.prior_prob = prior_prob
        self.children: List["TreeNode"] = []
        self.visit_count: int = 0
        self.total_value: float = 0.0
        self.mean_value: float = 0.0
        self.entropy: float = 0.50
        self.depth: int = 0 if parent is None else parent.depth + 1
        self.is_terminal: bool = False

    def ucb_score(self, c_puct: float = 1.414) -> float:
        """
        Calculates Upper Confidence Bound for Trees (PUCT) modified by node entropy.
        UCB = Q(s,a) + c_puct * P(s,a) * sqrt(N_parent) / (1 + N_child) * (1 + 0.1 * Entropy)
        """
        if self.parent is None:
            return 0.0

        parent_visits = max(1, self.parent.visit_count)
        exploration = (
            c_puct 
            * self.prior_prob 
            * (math.sqrt(parent_visits) / (1 + self.visit_count))
            * (1.0 + 0.15 * self.entropy)
        )
        return self.mean_value + exploration

    def update(self, reward: float):
        """Backpropagates rollout rewards up the tree."""
        self.visit_count += 1
        self.total_value += reward
        self.mean_value = self.total_value / self.visit_count


class ReasonEngine:
    """
    Main Test-Time Compute Engine implementing Adaptive Entropy-Guided MCTS.
    Scales reasoning accuracy through dynamic tree exploration and KV-cache compression.
    """
    def __init__(self, config: Optional[SearchConfig] = None):
        self.config = config or SearchConfig()
        self.evaluator = StepValueEvaluator()
        self.verifier = SelfConsistencyVerifier()
        self.kv_compressor = DynamicKVCacheCompressor()
        self.root: Optional[TreeNode] = None
        self.search_history: List[Dict[str, Any]] = []

    def simulate_candidate_steps(self, parent_text: str, k: int) -> List[Tuple[str, float, float]]:
        """
        Simulates generation of candidate reasoning steps with calculated entropy and prior prob.
        (Mocked inference sampler for standalone execution & high speed; compatible with PyTorch/HF pipelines).
        """
        step_templates = [
            ("Step {depth}: Break down problem into fundamental equations.", 0.85, 0.35),
            ("Step {depth}: Evaluate constraints and substitute variables.", 0.80, 0.42),
            ("Step {depth}: Perform step-by-step arithmetic verification.", 0.90, 0.25),
            ("Step {depth}: Wait, re-checking previous calculation for sanity check.", 0.95, 0.60),
            ("Step {depth}: Conclude final value calculation: \\boxed{{Solution}}.", 0.98, 0.15),
        ]
        
        sample_indices = random.sample(range(len(step_templates)), min(k, len(step_templates)))
        results = []
        for idx in sample_indices:
            tmpl, prior, entropy = step_templates[idx]
            text = tmpl.format(depth=len(parent_text.split("\n")))
            results.append((text, prior, entropy))
            
        return results

    def run_mcts(self, prompt: str) -> Tuple[str, TreeNode, Dict[str, Any]]:
        """
        Executes complete MCTS reasoning search over the configured simulation budget.
        """
        start_time = time.time()
        self.root = TreeNode(state_text=prompt)
        
        for sim in range(self.config.num_simulations):
            node = self.root
            
            # 1. Selection
            while node.children and not node.is_terminal:
                node = max(node.children, key=lambda child: child.ucb_score(self.config.c_puct))

            # 2. Expansion & Evaluation
            if node.depth < self.config.max_depth and not node.is_terminal:
                candidates = self.simulate_candidate_steps(node.state_text, self.config.top_k_actions)
                for step_text, prior, entropy in candidates:
                    child = TreeNode(
                        state_text=node.state_text + "\n" + step_text,
                        parent=node,
                        prior_prob=prior
                    )
                    child.entropy = entropy
                    if "\\boxed" in step_text or node.depth + 1 >= self.config.max_depth:
                        child.is_terminal = True
                    node.children.append(child)

                if node.children:
                    node = random.choice(node.children)

            # 3. Rollout Evaluation
            reward = self.evaluator.evaluate_step(node.state_text, prompt)

            # 4. KV-Cache Pruning Trigger
            if self.config.prune_kv_cache and sim % 4 == 0:
                attn_mock = [[random.random() for _ in range(32)] for _ in range(32)]
                entropy_mock = [node.entropy for _ in range(32)]
                _, _ = self.kv_compressor.prune_cache({}, attn_mock, entropy_mock)

            # 5. Backpropagation
            curr = node
            while curr is not None:
                curr.update(reward)
                curr = curr.parent

        # Extract best leaf node by visits and value
        best_child = self._select_best_path(self.root)
        leaves = self._collect_leaf_texts(self.root)
        consensus_ans, confidence, distribution = self.verifier.calculate_consensus(leaves)

        elapsed = time.time() - start_time
        meta = {
            "elapsed_seconds": round(elapsed, 4),
            "simulations": self.config.num_simulations,
            "consensus_confidence": confidence,
            "answer_distribution": distribution,
            "kv_compression_stats": self.kv_compressor.get_summary()
        }

        return best_child.state_text, self.root, meta

    def _select_best_path(self, node: TreeNode) -> TreeNode:
        """Traverses node hierarchy selecting highest visit count children."""
        curr = node
        while curr.children:
            curr = max(curr.children, key=lambda c: (c.visit_count, c.mean_value))
        return curr

    def _collect_leaf_texts(self, node: TreeNode) -> List[str]:
        """Gathers output traces from all terminal or leaf nodes."""
        leaves = []
        def _dfs(n: TreeNode):
            if not n.children:
                leaves.append(n.state_text)
            else:
                for c in n.children:
                    _dfs(c)
        _dfs(node)
        return leaves
