"""
Comprehensive Unit Test Suite for HyperReason Engine
Tests MCTS engine, KV compressor, FlashKV zero-copy manager, speculative tree engine, datasets, and wrap_model API.
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
    OllamaModelAdapter,
    FlashKVTreeManager,
    SpeculativeTreeEngine,
    wrap_model
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

    def test_flash_kv_zero_copy_manager(self):
        flash_kv = FlashKVTreeManager(block_size=16)
        root_blocks = flash_kv.allocate_root_blocks(node_id=0, seq_len=64)
        self.assertEqual(len(root_blocks), 4)

        child_blocks = flash_kv.branch_child_node(parent_node_id=0, child_node_id=1, new_tokens_count=16)
        self.assertEqual(len(child_blocks), 5)

        stats = flash_kv.get_memory_stats()
        self.assertGreater(stats["saved_vram_mb"], 0.0)

    def test_speculative_tree_engine(self):
        spec_engine = SpeculativeTreeEngine()
        branches = spec_engine.generate_speculative_branches("Step 1: Test step state", num_branches=4)
        self.assertEqual(len(branches), 4)
        
        stats = spec_engine.get_speculative_stats()
        self.assertIn("acceptance_rate_pct", stats)

    def test_wrap_model_api(self):
        dummy_model = "dummy_llm"
        wrapped = wrap_model(dummy_model)
        res = wrapped.reason("If 4 workers build a wall in 6 hours, how many hours for 8 workers?")
        
        self.assertIn("solution_trajectory", res)
        self.assertIn("flash_kv_stats", res["metrics"])

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
