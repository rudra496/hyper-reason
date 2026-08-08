"""
Rich ASCII Terminal Visualizer for Reasoning Trees
Renders dynamic MCTS reasoning rollout graphs directly in terminal CLI.
"""

from typing import List, Optional
from .mcts_engine import TreeNode


class TreeVisualizer:
    """
    Renders ASCII tree visual representation of the Monte Carlo reasoning search tree.
    """
    def __init__(self, max_render_depth: int = 4):
        self.max_render_depth = max_render_depth

    def render(self, root: TreeNode, highlight_best_path: bool = True) -> str:
        """Generates formatted multi-line ASCII tree diagram."""
        lines = []
        lines.append("⚡ HyperReason Engine — Monte Carlo Tree Search Visualization ⚡")
        lines.append("=" * 65)

        def _build_tree(node: TreeNode, prefix: str = "", is_last: bool = True):
            if node.depth > self.max_render_depth:
                return

            connector = "└── " if is_last else "├── "
            node_label = f"[Depth {node.depth}] N={node.visit_count} | Q={node.mean_value:.3f} | H={node.entropy:.2f}"
            
            if node.depth == 0:
                lines.append(f"ROOT: {node.state_text[:40]}...")
            else:
                last_line = node.state_text.split("\n")[-1] if "\n" in node.state_text else node.state_text
                lines.append(f"{prefix}{connector}{node_label} -> \"{last_line[:35]}...\"")

            child_prefix = prefix + ("    " if is_last else "│   ")
            sorted_children = sorted(node.children, key=lambda c: c.visit_count, reverse=True)
            for i, child in enumerate(sorted_children[:3]):  # limit rendering to top 3 branches per node
                _build_tree(child, child_prefix, i == len(sorted_children[:3]) - 1)

        _build_tree(root)
        lines.append("=" * 65)
        return "\n".join(lines)
