"""
Comprehensive Unit Test Suite for HyperReason Engine
Tests MCTS engine, KV compressor, FlashKV zero-copy manager, speculative tree engine, trace exporters, search presets, cost analyzers, quantizers, multi-agent trees, and HTML visualizers.
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
    wrap_model,
    TreeTraceExporter,
    SearchPresets,
    CostEfficiencyAnalyzer,
    KVQuantizer,
    MultiAgentReasonTree,
    HTMLTreeVisualizer
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

    def test_kv_quantizer(self):
        quantizer = KVQuantizer(quant_bits=8)
        quantized, scale, zero_pt = quantizer.quantize_values([0.5, -0.2, 0.9, -0.8])
        self.assertEqual(len(quantized), 4)
        
        stats = quantizer.get_quantization_stats(original_count=1024)
        self.assertEqual(stats["quantization_bits"], 8)

    def test_multi_agent_tree(self):
        agent_tree = MultiAgentReasonTree()
        res = agent_tree.run_collaborative_mcts("Solve 2 + 2")
        self.assertIn("multi_agent_collaboration", res["metrics"])

    def test_html_tree_visualizer(self):
        root = TreeNode("Root state")
        child = TreeNode("Child state", parent=root)
        root.children.append(child)

        html_vis = HTMLTreeVisualizer(root)
        html_str = html_vis.export_html()
        self.assertIn("HyperReason Interactive Search Tree", html_str)

    def test_wrap_model_api(self):
        dummy_model = "dummy_llm"
        wrapped = wrap_model(dummy_model)
        res = wrapped.reason("If 4 workers build a wall in 6 hours, how many hours for 8 workers?")
        
        self.assertIn("solution_trajectory", res)
        self.assertIn("flash_kv_stats", res["metrics"])

    def test_trace_exporters(self):
        root = TreeNode("Root state")
        child = TreeNode("Child state", parent=root)
        root.children.append(child)

        exporter = TreeTraceExporter(root)
        json_str = exporter.to_json()
        md_str = exporter.to_markdown()

        self.assertIn("Root state", json_str)
        self.assertIn("HyperReason Solution Trajectory", md_str)

    def test_search_presets(self):
        fast_cfg = SearchPresets.ultra_fast()
        high_cfg = SearchPresets.high_accuracy()

        self.assertEqual(fast_cfg.num_simulations, 12)
        self.assertEqual(high_cfg.num_simulations, 64)

    def test_cost_analyzer(self):
        analyzer = CostEfficiencyAnalyzer()
        res = analyzer.analyze(num_queries=1000, avg_vram_saved_mb=2048.0, latency_reduction_pct=50.0)

        self.assertEqual(res["query_volume"], 1000)
        self.assertIn("$", res["estimated_cloud_dollars_saved"])

    def test_gsm8k_dataset_loader(self):
        dataset = GSM8KDataset().get_problems()
        self.assertGreater(len(dataset), 0)

    def test_vllm_paged_attention_hook(self):
        hook = VLLMPagedAttentionHook()
        table = [1, 2, 3, 4, 5, 6, 7, 8]
        retained, stats = hook.prune_paged_kv_blocks(table, [0.5]*8, 0.5)
        self.assertLessEqual(len(retained), len(table))


if __name__ == "__main__":
    unittest.main()
