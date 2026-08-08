"""
Unit Test Suite for HyperReason Library
Validates MCTS search, UCB scoring, KV cache pruning, and consensus verifiers.
"""

import unittest
from hyper_reason import (
    ReasonEngine,
    SearchConfig,
    TreeNode,
    DynamicKVCacheCompressor,
    StepValueEvaluator,
    SelfConsistencyVerifier
)


class TestHyperReason(unittest.TestCase):

    def test_tree_node_ucb(self):
        root = TreeNode(state_text="Root")
        root.visit_count = 10
        child = TreeNode(state_text="Child", parent=root, prior_prob=0.8)
        child.entropy = 0.4
        child.update(0.75)
        
        score = child.ucb_score(c_puct=1.414)
        self.assertGreater(score, 0.0)

    def test_kv_compressor_pruning(self):
        compressor = DynamicKVCacheCompressor()
        mock_attn = [[0.1 * ((i + j) % 10) for j in range(40)] for i in range(40)]
        mock_entropy = [0.3 for _ in range(40)]
        
        pruned_cache, stats = compressor.prune_cache({}, mock_attn, mock_entropy)
        self.assertIn("pruned_tokens", stats)
        self.assertGreaterEqual(stats["pruned_tokens"], 0)

    def test_evaluator_step_rewards(self):
        evaluator = StepValueEvaluator()
        good_step = "Therefore, we evaluate x = 12 * 3 = 36."
        bad_step = "I cannot answer this question."
        
        good_score = evaluator.evaluate_step(good_step, "prompt")
        bad_score = evaluator.evaluate_step(bad_step, "prompt")
        
        self.assertGreater(good_score, bad_score)

    def test_reason_engine_mcts_run(self):
        config = SearchConfig(num_simulations=8, max_depth=3)
        engine = ReasonEngine(config=config)
        trace, root, meta = engine.run_mcts("Test math problem 2 + 2")
        
        self.assertIsNotNone(root)
        self.assertIn("consensus_confidence", meta)
        self.assertTrue(len(trace) > 0)


if __name__ == "__main__":
    unittest.main()
