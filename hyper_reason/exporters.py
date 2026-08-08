"""
Trace Exporters & Visualization Formatter Module
Author: Rudra Sarker & Buggz
License: MIT

Exports MCTS reasoning tree trajectories to JSON, Markdown, LaTeX, or interactive HTML formats.
"""

import json
from typing import Dict, Any, List, Optional
from .mcts_engine import TreeNode


class TreeTraceExporter:
    """
    Exports MCTS reasoning search trees into standardized external formats.
    """
    def __init__(self, root: TreeNode):
        self.root = root

    def to_dict(self, node: Optional[TreeNode] = None) -> Dict[str, Any]:
        """Converts search tree hierarchy to nested dictionary object."""
        curr = node or self.root
        return {
            "depth": curr.depth,
            "state_text": curr.state_text,
            "visit_count": curr.visit_count,
            "mean_value": round(curr.mean_value, 4),
            "entropy": round(curr.entropy, 4),
            "is_terminal": curr.is_terminal,
            "children": [self.to_dict(child) for child in curr.children]
        }

    def to_json(self, indent: int = 2) -> str:
        """Exports search tree as formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def to_markdown(self) -> str:
        """Exports best reasoning path as GitHub-Flavored Markdown document."""
        root_first_line = self.root.state_text.splitlines()[0] if self.root.state_text else ""
        lines = ["# HyperReason Solution Trajectory\n"]
        lines.append(f"**Root Prompt**: {root_first_line}\n")
        lines.append("## Reasoning Search Steps:\n")

        curr = self.root
        step_idx = 1
        while curr.children:
            curr = max(curr.children, key=lambda c: c.visit_count)
            lines.append(f"### Step {step_idx} (Visits: {curr.visit_count}, Value: {curr.mean_value:.3f}, Entropy: {curr.entropy:.2f})")
            last_line = curr.state_text.splitlines()[-1] if curr.state_text else ""
            lines.append(f"```text\n{last_line}\n```\n")
            step_idx += 1

        return "\n".join(lines)
