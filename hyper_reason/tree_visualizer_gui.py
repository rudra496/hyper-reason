"""
Interactive SVG/HTML Search Graph Generator
Author: Rudra Sarker & Buggz
License: MIT

Generates standalone interactive HTML visual search tree diagrams of MCTS rollout trajectories.
"""

from typing import Optional
from .mcts_engine import TreeNode


class HTMLTreeVisualizer:
    """
    Generates interactive HTML search tree visualization files.
    """
    def __init__(self, root: TreeNode):
        self.root = root

    def export_html(self, output_path: Optional[str] = None) -> str:
        """Generates self-contained HTML page displaying interactive search tree graph."""
        nodes_data = []
        
        def _traverse(node: TreeNode, parent_id: Optional[int] = None, current_id: int = 0) -> int:
            last_line = node.state_text.split("\n")[-1] if "\n" in node.state_text else node.state_text
            nodes_data.append({
                "id": current_id,
                "parent_id": parent_id,
                "label": f"Depth {node.depth} | N={node.visit_count} | Q={node.mean_value:.2f}",
                "text": last_line[:40]
            })
            
            next_id = current_id + 1
            for child in node.children[:3]:  # limit to top 3 branches per node
                next_id = _traverse(child, current_id, next_id)
            return next_id

        _traverse(self.root)

        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>HyperReason Interactive Search Tree</title>
    <style>
        body {{ background: #0a0c10; color: #f0f6fc; font-family: sans-serif; padding: 2rem; }}
        .tree-node {{ background: #12161f; border: 1px solid #30363d; padding: 1rem; border-radius: 8px; margin: 10px 0; }}
        .node-title {{ color: #00f2fe; font-weight: bold; }}
    </style>
</head>
<body>
    <h1>⚡ HyperReason Tree Search Rollout Graph</h1>
    <div id="tree">
"""
        for n in nodes_data:
            indent = "&nbsp;" * (n["id"] * 4)
            html_content += f'<div class="tree-node">{indent}<span class="node-title">{n["label"]}</span>: "{n["text"]}"</div>\n'

        html_content += "</div></body></html>"

        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(html_content)

        return html_content
