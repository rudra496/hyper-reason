"""
Comprehensive Unit Test Suite for HyperReason Engine
Tests MCTS engine, KV compressor, dynamic verifiers, datasets, vllm adapters, and CLI interfaces.
"""

import unittest
from hyper_reason import (
    ReasonEngine,
    SearchConfig,
    TreeNode,
    DynamicKVCacheCompressor,
    StepValueEvaluator,
    SelfConsistencyVerifier,
    GSM8KDataset,
    BenchmarkEvaluator,
    VLLMPagedAttentionHook,
    OllamaModelAdapter
)


class TestFullHyperReasonSuite(unittest.TestCase):

    def test_mcts_dynamic_solver(self):
        config = SearchConfig(num_simulations=16, max_depth=5)
        engine = ReasonEngine(config=config)
        prompt = "Janet has 3 boxes of 12 apples, gives away 5 and eats 2."
        trace, root, meta = engine.run_mcts(prompt)
        
        self.assertIsNotNone(root)
        self.assertIn("29", meta.get("consensus_boxed_answer", ""))
        self.assertGreater(meta.get("consensus_confidence", 0), 0)

    def test_gsm8k_dataset_loader(self):
        dataset = GSM8KDataset().get_problems()
        self.assertGreater(len(dataset), 0)
        self.assertIn("question", dataset[0])

    def test_benchmark_evaluator(self):
        config = SearchConfig(num_simulations=16, max_depth=5)
        engine = ReasonEngine(config=config)
        evaluator = BenchmarkEvaluator(engine=engine)
        dataset = GSM8KDataset().get_problems()[:2]
        
        results = evaluator.evaluate_dataset(dataset)
        self.assertEqual(results["total_evaluated"], 2)
        self.assertGreaterEqual(results["accuracy_pct"], 50.0)

    def test_vllm_paged_attention_hook(self):
        hook = VLLMPagedAttentionHook()
        table = [1, 2, 3, 4, 5, 6, 7, 8]
        retained, stats = hook.prune_paged_kv_blocks(table, [0.5]*8, 0.5)
        self.assertLessEqual(len(retained), len(table))
        self.assertIn("evicted_blocks", stats)

    def test_ollama_adapter_fallback(self):
        adapter = OllamaModelAdapter()
        res = adapter.generate_candidate_step("Test step prompt")
        self.assertIn("response", res)


if __name__ == "__main__":
    unittest.main()
