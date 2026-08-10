"""Adaptive Entropy-Guided MCTS — the REAL engine.

Unlike the v1.x fake (which regex-solved arithmetic and never called a model), this engine
genuinely asks a backend for K candidate continuations at each expansion, derives
**sample-diversity entropy** and honest priors from those samples, searches with AE-PUCT, and
picks the final answer by **self-consistency** over the terminal trajectories the model
actually produced. No hardcoded answers, no word-matching rewards, no random KV.

Budget is bounded and explicit: ``num_simulations`` expansions, each sampling ``k_samples``
candidates -> total model calls == num_simulations * k_samples (plus a hard cap).
"""

from __future__ import annotations

import math
import time
from typing import Any, Optional

from ..backends.base import ModelBackend, count_tokens
from .config import SearchConfig
from .entropy import sample_diversity_entropy, priors_from_diversity
from .math_utils import extract_final_answer
from .verifier import self_consistency

_PROMPT_TEMPLATE = (
    "Solve the problem step by step. Show each reasoning step on its own line.\n"
    "At the end, put ONLY the final numerical answer in \\boxed{{}}.\n\n"
    "Problem: {problem}\n"
)


class TreeNode:
    """A node in the reasoning tree. Field surface is kept compatible with the salvaged
    exporters/visualizers (state_text, parent, children, visit_count, mean_value, depth, entropy).
    """

    def __init__(self, state_text: str, parent: Optional["TreeNode"] = None, prior_prob: float = 1.0):
        self.state_text = state_text
        self.parent = parent
        self.prior_prob = prior_prob
        self.children: list["TreeNode"] = []
        self.visit_count = 0
        self.total_value = 0.0
        self.mean_value = 0.0
        self.entropy = 0.0  # sample-diversity entropy of the candidates that created this node
        self.depth = 0 if parent is None else parent.depth + 1
        self.is_terminal = False
        self.expanded = False
        self.kv_tokens = count_tokens(state_text)  # real token count -> honest KV length
        self.completion_tokens = 0
        self._value_cache: Optional[float] = None

    def ucb_score(self, c_puct: float, entropy_alpha: float) -> float:
        """AE-PUCT: mean_value + c_puct*prior*sqrt(Np)/(1+Nc) * (1 + alpha*diversity_entropy)."""
        if self.parent is None:
            return 0.0
        parent_visits = max(1, self.parent.visit_count)
        exploration = (
            c_puct
            * self.prior_prob
            * (math.sqrt(parent_visits) / (1 + self.visit_count))
            * (1.0 + entropy_alpha * self.entropy)
        )
        return self.mean_value + exploration

    def update(self, value: float) -> None:
        self.visit_count += 1
        self.total_value += value
        self.mean_value = self.total_value / self.visit_count


class ReasonEngine:
    """Real AE-MCTS over a model backend."""

    def __init__(self, backend: ModelBackend, config: Optional[SearchConfig] = None):
        if not isinstance(backend, ModelBackend):
            raise TypeError("ReasonEngine needs a ModelBackend (e.g. GLMBackend() or MockBackend()).")
        self.backend = backend
        self.config = config or SearchConfig()
        self.root: Optional[TreeNode] = None
        self.flashkv: Any = None  # optional projector (Phase 3); None -> no projected stats
        self._model_calls = 0

    def attach_flashkv(self, flashkv: Any) -> None:
        """Attach a FlashKV projector (Phase 3). It reads real per-node token counts."""
        self.flashkv = flashkv

    # -- public API ---------------------------------------------------------
    def reason(self, problem: str) -> dict:
        cfg = self.config
        base_prompt = _PROMPT_TEMPLATE.format(problem=problem)
        self.root = TreeNode(base_prompt)

        max_calls = cfg.num_simulations * cfg.k_samples  # hard budget
        sims_executed = 0
        depth_reached = 0
        total_prompt_tokens = 0
        total_completion_tokens = 0
        diversity_entropies: list[float] = []
        t0 = time.time()

        for _ in range(cfg.num_simulations):
            if self._model_calls >= max_calls:
                break
            node, path = self._select(self.root)
            if node is None:
                break  # no expandable node remains
            sims_executed += 1
            if (not node.expanded) and (not node.is_terminal) and (node.depth < cfg.max_depth):
                value, ptoks, ctoks, dent = self._expand(node)
                total_prompt_tokens += ptoks
                total_completion_tokens += ctoks
                diversity_entropies.append(dent)
                depth_reached = max(depth_reached, node.depth + 1)
            else:
                value = self._terminal_value(node)
            for n in path:
                n.update(value)

        elapsed = time.time() - t0

        # Final answer = self-consistency over all terminal trajectories the model produced.
        terminals = self._collect_terminals(self.root)
        traces = [t.state_text for t in terminals]
        answer, confidence, distribution = self_consistency(traces)
        best_trace = self._best_trace_for(answer, terminals) or (self.root.state_text if self.root else "")

        flashkv_stats = {}
        if self.flashkv is not None:
            flashkv_stats = self.flashkv.project(self.root)

        return {
            "prompt": problem,
            "solution_trajectory": best_trace,
            "boxed_answer": answer,
            "confidence": confidence,
            "metrics": {
                "elapsed_seconds": round(elapsed, 4),
                "simulations_executed": sims_executed,
                "model_calls": self._model_calls,
                "max_depth_reached": depth_reached,
                "total_prompt_tokens": total_prompt_tokens,
                "total_completion_tokens": total_completion_tokens,
                "diversity_entropy_mean": (
                    round(sum(diversity_entropies) / len(diversity_entropies), 4) if diversity_entropies else 0.0
                ),
                "num_terminal_answers": len(traces),
                "sc_distribution": distribution,
                "backend": self.backend.name,
                "backend_is_live": self.backend.is_live,
                "config": cfg.disclosure(),
                "flashkv": flashkv_stats,
            },
            "tree": self.root,
        }

    # -- internals ----------------------------------------------------------
    def _select(self, root: TreeNode) -> tuple[Optional[TreeNode], list[TreeNode]]:
        """Descend by AE-PUCT to the next expandable (un-expanded, non-terminal, < max_depth) node."""
        cfg = self.config
        node = root
        path = [root]
        if (not root.expanded) and (not root.is_terminal) and (root.depth < cfg.max_depth):
            return root, path
        guard = 0
        while True:
            guard += 1
            if node.is_terminal or node.depth >= cfg.max_depth or not node.children:
                return None, path
            kids = [c for c in node.children if not c.is_terminal]
            if not kids:
                return None, path
            unexp = [c for c in kids if not c.expanded]
            choose_from = unexp if unexp else kids
            node = max(choose_from, key=lambda c: c.ucb_score(cfg.c_puct, cfg.entropy_alpha))
            path.append(node)
            if (not node.expanded) and (not node.is_terminal) and (node.depth < cfg.max_depth):
                return node, path
            if guard > cfg.max_depth + 4:  # safety; should not trigger
                return None, path

    def _expand(self, node: TreeNode) -> tuple[float, int, int, float]:
        """Sample K real continuations, create children, return (value, ptoks, ctoks, entropy)."""
        cfg = self.config
        node.expanded = True
        samples = self.backend.sample(
            node.state_text,
            k=cfg.k_samples,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens_per_step,
            stop=(),
        )
        self._model_calls += len(samples)
        texts = [s.text for s in samples if s.text]
        entropy = sample_diversity_entropy(texts)
        priors = priors_from_diversity(texts)
        node.entropy = entropy
        ptoks = sum(getattr(s, "prompt_tokens", 0) for s in samples)
        ctoks = sum(getattr(s, "completion_tokens", 0) for s in samples)
        max_terminal_value = 0.0
        for i, s in enumerate(samples):
            child = TreeNode(
                state_text=node.state_text + "\n" + s.text,
                parent=node,
                prior_prob=priors[i] if i < len(priors) else 1.0 / max(1, len(samples)),
            )
            child.completion_tokens = getattr(s, "completion_tokens", 0)
            child.is_terminal = ("\\boxed" in s.text) or (child.depth >= cfg.max_depth)
            if child.is_terminal:
                v = 1.0 if extract_final_answer(child.state_text) != "__unparsable__" else 0.0
                child._value_cache = v
                max_terminal_value = max(max_terminal_value, v)
            node.children.append(child)
        return max_terminal_value, ptoks, ctoks, entropy

    def _terminal_value(self, node: TreeNode) -> float:
        if node._value_cache is not None:
            return node._value_cache
        if node.is_terminal:
            v = 1.0 if extract_final_answer(node.state_text) != "__unparsable__" else 0.0
            node._value_cache = v
            return v
        return 0.0

    def _collect_terminals(self, root: TreeNode) -> list[TreeNode]:
        out: list[TreeNode] = []
        stack = [root]
        while stack:
            n = stack.pop()
            if n.is_terminal:
                out.append(n)
            stack.extend(n.children)
        return out

    def _best_trace_for(self, answer: str, terminals: list[TreeNode]) -> Optional[str]:
        matches = [t for t in terminals if extract_final_answer(t.state_text) == answer]
        if not matches and terminals:
            matches = terminals
        if not matches:
            return None
        best = max(matches, key=lambda t: (t.visit_count, t.mean_value))
        return best.state_text
