"""Engine tests — the real AE-MCTS, on Mock (deterministic) and live GLM (real proof).

These assert genuine structural/search properties, not hardcoded answers.
"""

import os
import math
import pytest

from hyper_reason import (
    wrap_model, ReasonEngine, TreeNode, SearchConfig, SearchPresets,
    TreeTraceExporter, TreeVisualizer,
)
from hyper_reason.backends import MockBackend, GLMBackend
from hyper_reason.engine.mcts import _PROMPT_TEMPLATE  # private but stable enough for a check


PROBLEM = "Janet has 3 boxes of 12 apples. She gives away 5 and eats 2. How many apples remain?"


class TestRealEngineOnMock:
    def test_wrap_model_none_uses_mock(self):
        m = wrap_model()
        assert m.backend.is_live is False
        assert "mock" in m.backend.name

    def test_wrap_model_rejects_non_backend(self):
        with pytest.raises(TypeError):
            wrap_model("dummy_llm")  # the v1.x "test" that passed with a fake engine must now FAIL

    def test_reason_returns_real_search_structure(self):
        engine = ReasonEngine(MockBackend(), SearchPresets.balanced())
        res = engine.reason(PROBLEM)
        m = res["metrics"]
        assert res["boxed_answer"]  # some answer extracted
        assert 0.0 <= res["confidence"] <= 1.0
        # Bounded, real budget: every expansion sampled exactly k candidates.
        assert m["model_calls"] == m["simulations_executed"] * SearchPresets.balanced().k_samples
        assert m["model_calls"] <= SearchPresets.balanced().num_simulations * SearchPresets.balanced().k_samples
        assert m["backend_is_live"] is False
        assert "sample_diversity_entropy" in m["config"]["entropy_source"]

    def test_tree_genuinely_explores(self):
        engine = ReasonEngine(MockBackend(), SearchConfig(num_simulations=9, max_depth=4, k_samples=3))
        res = engine.reason(PROBLEM)
        root = res["tree"]
        assert root.visit_count >= 1
        # Some expansions happened -> there are children with visits.
        total_children = sum(1 for _ in _walk(root)) - 1
        assert total_children > 0
        visited = [n.visit_count for n in _walk(root)]
        assert max(visited) >= 1
        assert m_depth(root) >= 1

    def test_terminal_collection_and_sc(self):
        engine = ReasonEngine(MockBackend(), SearchConfig(num_simulations=12, max_depth=3, k_samples=3))
        res = engine.reason(PROBLEM)
        # At max_depth every node is terminal -> SC distribution must be non-empty.
        assert res["metrics"]["num_terminal_answers"] > 0
        assert res["metrics"]["sc_distribution"]  # a real vote happened

    def test_exporters_and_visualizer_work_on_real_tree(self):
        res = ReasonEngine(MockBackend(), SearchPresets.ultra_fast()).reason(PROBLEM)
        root = res["tree"]
        j = TreeTraceExporter(root).to_json()
        assert "state_text" in j
        md = TreeTraceExporter(root).to_markdown()
        assert "HyperReason Solution Trajectory" in md
        ascii_tree = TreeVisualizer().render(root)
        assert "HyperReason Engine" in ascii_tree

    def test_treenode_ae_puct(self):
        parent = TreeNode("p")
        child = TreeNode("c", parent=parent, prior_prob=0.5)
        child.entropy = 1.0
        parent.update(1.0)
        child.update(0.5)
        # AE-PUCT must be > mean_value (exploration term strictly positive with entropy>0).
        assert child.ucb_score(c_puct=1.414, entropy_alpha=0.15) > child.mean_value


class TestLiveGLEngine:
    KEY = os.environ.get("ANTHROPIC_API_KEY")

    @pytest.mark.skipif(not KEY, reason="no ANTHROPIC_API_KEY in env")
    def test_real_model_generates_real_trajectory(self):
        # END-TO-END PROOF: a real model drives the search and produces model output (not a template).
        cfg = SearchPresets.ultra_fast()  # sims=6, k=2, depth=3 -> ~12 GLM calls
        res = wrap_model(GLMBackend(model="glm-4.6"), config=cfg).reason("What is 17 + 25?")
        m = res["metrics"]
        assert m["backend_is_live"] is True
        assert m["model_calls"] == m["simulations_executed"] * cfg.k_samples
        # The trajectory is GLM-generated, not the Mock heuristic template.
        assert "intermediate =" not in res["solution_trajectory"]
        assert res["boxed_answer"] not in ("", "__unparsable__")


def _walk(node):
    yield node
    for c in node.children:
        yield from _walk(c)


def m_depth(node) -> int:
    return max((n.depth for n in _walk(node)), default=0)
