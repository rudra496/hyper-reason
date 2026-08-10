"""
HyperReason Demo: Test-Time Compute Reasoning on GSM8K Math Problems (v2)
Demonstrates programmatic API usage of ReasonEngine and TreeVisualizer.
"""

import os
import json
from hyper_reason import ReasonEngine, SearchConfig, TreeVisualizer, GLMBackend, MockBackend

def run_demo():
    print("=" * 70)
    print("      HYPERREASON ENGINE: RUNNING TEST-TIME REASONING DEMO           ")
    print("=" * 70)

    problem_prompt = (
        "Janet has 3 boxes of apples. Each box contains 12 apples. "
        "She gives away 5 apples to her neighbor and eats 2. How many apples does she have left?"
    )

    print(f"Problem: {problem_prompt}\n")

    # Configure engine for search
    config = SearchConfig(
        num_simulations=12,
        max_depth=4,
        k_samples=2,
        c_puct=1.414,
    )

    backend = GLMBackend() if os.environ.get("ANTHROPIC_API_KEY") else MockBackend()
    engine = ReasonEngine(backend, config)

    res = engine.reason(problem_prompt)

    # Render tree visualizer
    if "root" in res and res["root"]:
        visualizer = TreeVisualizer(max_render_depth=3)
        print(visualizer.render(res["root"]))

    print("\n💡 Final Solution Trajectory:")
    print(f"Boxed Answer: {res.get('boxed_answer')}")
    print(f"Confidence: {res.get('confidence')}")

    print("\n⚡ Execution Metrics:")
    print(json.dumps(res.get("metrics", {}), indent=2))

if __name__ == "__main__":
    run_demo()
