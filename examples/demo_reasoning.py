"""
HyperReason Demo: Test-Time Compute Reasoning on GSM8K Math Problems
Demonstrates programmatic API usage of ReasonEngine and DynamicKVCacheCompressor.
"""

from hyper_reason import ReasonEngine, SearchConfig, TreeVisualizer

def run_demo():
    print("=" * 70)
    print("      HYPERREASON ENGINE: RUNNING TEST-TIME REASONING DEMO           ")
    print("=" * 70)

    problem_prompt = (
        "Question: Janet has 3 boxes of apples. Each box contains 12 apples. "
        "She gives away 5 apples to her neighbor and eats 2. How many apples does she have left?"
    )

    print(f"Problem: {problem_prompt}\n")

    # Configure engine for high accuracy search
    config = SearchConfig(
        num_simulations=24,
        max_depth=5,
        c_puct=1.414,
        prune_kv_cache=True
    )

    engine = ReasonEngine(config=config)
    best_trajectory, root_node, metrics = engine.run_mcts(problem_prompt)

    # Render tree visualizer
    visualizer = TreeVisualizer(max_render_depth=3)
    print(visualizer.render(root_node))

    print("\n💡 Final Solution Trajectory:")
    print(best_trajectory)

    print("\n⚡ Dynamic KV-Cache Compression Summary:")
    stats = metrics["kv_compression_stats"]
    print(f"  • Total Tokens Evaluated: {stats['total_processed']}")
    print(f"  • Tokens Pruned: {stats['total_pruned']}")
    print(f"  • VRAM Memory Saved: {stats['estimated_vram_saved_gb']} GB")
    print(f"  • Compression Efficiency: {stats['overall_compression_ratio']}%")

if __name__ == "__main__":
    run_demo()
