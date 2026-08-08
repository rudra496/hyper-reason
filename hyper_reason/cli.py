"""
HyperReason Command Line Interface (CLI)
CLI entrypoint for running test-time compute searches, tree visualizers, and benchmark suites.
"""

import sys
import argparse
import json
from hyper_reason import ReasonEngine, SearchConfig, TreeVisualizer

def main():
    parser = argparse.ArgumentParser(
        description="HyperReason CLI: Autonomous Test-Time Compute Scaling & Dynamic KV-Cache Sparsification"
    )
    parser.add_argument(
        "--prompt", "-p", 
        type=str, 
        default="Solve math problem: If a train travels at 60 mph for 2.5 hours, how far does it travel?",
        help="Input problem prompt for reasoning search."
    )
    parser.add_argument(
        "--simulations", "-s", 
        type=int, 
        default=32, 
        help="Number of MCTS rollout simulations (default: 32)."
    )
    parser.add_argument(
        "--depth", "-d", 
        type=int, 
        default=6, 
        help="Maximum depth of reasoning tree (default: 6)."
    )
    parser.add_argument(
        "--no-kv-pruning", 
        action="store_true", 
        help="Disable token-attentive KV-cache pruning."
    )
    parser.add_argument(
        "--visualize", "-v", 
        action="store_true", 
        default=True,
        help="Render ASCII tree search diagram in terminal."
    )

    args = parser.parse_args()

    print("🚀 Initializing HyperReason Engine v1.0.0...")
    print(f"📝 Prompt: {args.prompt}")
    print(f"⚙️ Config: Simulations={args.simulations}, Max Depth={args.depth}, KV Pruning={not args.no_kv_pruning}\n")

    config = SearchConfig(
        num_simulations=args.simulations,
        max_depth=args.depth,
        prune_kv_cache=not args.no_kv_pruning
    )

    engine = ReasonEngine(config=config)
    best_trace, root, meta = engine.run_mcts(args.prompt)

    if args.visualize and root is not None:
        visualizer = TreeVisualizer(max_render_depth=4)
        print(visualizer.render(root))
        print("\n")

    print("🏆 BEST REASONING TRAJECTORY FOUND:")
    print("-" * 50)
    print(best_trace)
    print("-" * 50)
    print("\n📊 SEARCH & COMPRESSION BENCHMARK METRICS:")
    print(json.dumps(meta, indent=2))

if __name__ == "__main__":
    main()
