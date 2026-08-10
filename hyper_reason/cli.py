"""Command-line interface for HyperReason (v2 — real engine)."""

from __future__ import annotations

import argparse
import sys

from .backends import MockBackend, GLMBackend, OllamaBackend
from .config_presets import SearchPresets
from .wrapper import wrap_model


def _backend_for(name: str):
    name = (name or "mock").lower()
    if name == "mock":
        return MockBackend()
    if name == "glm":
        return GLMBackend()
    if name == "ollama":
        return OllamaBackend()
    raise SystemExit(f"unknown backend '{name}' (choose: mock, glm, ollama)")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="hyper-reason", description="HyperReason AE-MCTS reasoner")
    p.add_argument("--problem", "-p", required=True, help="problem text to reason about")
    p.add_argument("--backend", "-b", default="mock", choices=["mock", "glm", "ollama"])
    p.add_argument("--preset", default="balanced",
                   choices=["ultra_fast", "balanced", "high_accuracy"])
    p.add_argument("--sims", type=int, default=None, help="override num_simulations")
    p.add_argument("--no-tree", action="store_true", help="don't print the ASCII tree")
    args = p.parse_args(argv)

    preset = {"ultra_fast": SearchPresets.ultra_fast, "balanced": SearchPresets.balanced,
              "high_accuracy": SearchPresets.high_accuracy}[args.preset]()
    if args.sims:
        preset.num_simulations = args.sims

    model = wrap_model(_backend_for(args.backend), config=preset)
    result = model.reason(args.problem)
    m = result["metrics"]

    print(f"\nBackend      : {m['backend']} (is_live={m['backend_is_live']})")
    print(f"Boxed answer : {result['boxed_answer']}")
    print(f"Confidence   : {result['confidence']}")
    print(f"Model calls  : {m['model_calls']}  depth≤{m['max_depth_reached']}  "
          f"tokens={m['total_prompt_tokens']}+{m['total_completion_tokens']}")
    print(f"Entropy(src) : {m['config']['entropy_source']}")
    print("\n--- Trajectory ---\n" + result["solution_trajectory"])

    if not args.no_tree:
        from .terminal_visualizer import TreeVisualizer
        print("\n" + TreeVisualizer().render(result["tree"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
