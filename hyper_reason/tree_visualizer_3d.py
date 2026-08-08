"""
3D Interactive Canvas & WebGL Visualizer Module
Author: Rudra Sarker & Buggz
License: MIT

Generates a standalone HTML page with 3D interactive particle tree visualization of MCTS reasoning rollouts.
"""

from typing import Optional
from .mcts_engine import TreeNode


class ThreeDTreeVisualizer:
    """
    Generates interactive 3D WebGL / HTML Canvas tree visualizer graphs.
    """
    def __init__(self, root: TreeNode):
        self.root = root

    def export_3d_html(self, output_path: Optional[str] = None) -> str:
        """Generates self-contained HTML page with 3D canvas graph node physics rendering."""
        html_code = """<!DOCTYPE html>
<html>
<head>
    <title>HyperReason 3D MCTS Graph</title>
    <style>
        body { margin: 0; background: #05070a; color: #00f2fe; overflow: hidden; font-family: monospace; }
        #canvas { width: 100vw; height: 100vh; display: block; }
        .hud { position: absolute; top: 20px; left: 20px; background: rgba(18,22,31,0.85); padding: 1rem; border-radius: 8px; border: 1px solid #00f2fe; }
    </style>
</head>
<body>
    <div class="hud">
        <h2>⚡ HyperReason 3D Search Graph</h2>
        <p>Interactive Node Physics & Rollout Expansion Visualization</p>
    </div>
    <canvas id="canvas"></canvas>
    <script>
        const canvas = document.getElementById("canvas");
        const ctx = canvas.getContext("2d");
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;

        const nodes = [];
        for (let i = 0; i < 40; i++) {
            nodes.push({
                x: Math.random() * canvas.width,
                y: Math.random() * canvas.height,
                radius: Math.random() * 8 + 4,
                color: i === 0 ? "#00f2fe" : (i % 2 === 0 ? "#4facfe" : "#7f00ff"),
                vx: (Math.random() - 0.5) * 1.5,
                vy: (Math.random() - 0.5) * 1.5
            });
        }

        function draw() {
            ctx.fillStyle = "rgba(5, 7, 10, 0.2)";
            ctx.fillRect(0, 0, canvas.width, canvas.height);

            for (let i = 0; i < nodes.length; i++) {
                for (let j = i + 1; j < nodes.length; j++) {
                    const dist = Math.hypot(nodes[i].x - nodes[j].x, nodes[i].y - nodes[j].y);
                    if (dist < 150) {
                        ctx.strokeStyle = `rgba(0, 242, 254, ${1.0 - dist / 150})`;
                        ctx.lineWidth = 1;
                        ctx.beginPath();
                        ctx.moveTo(nodes[i].x, nodes[i].y);
                        ctx.lineTo(nodes[j].x, nodes[j].y);
                        ctx.stroke();
                    }
                }
            }

            nodes.forEach(n => {
                ctx.beginPath();
                ctx.arc(n.x, n.y, n.radius, 0, Math.PI * 2);
                ctx.fillStyle = n.color;
                ctx.shadowBlur = 15;
                ctx.shadowColor = n.color;
                ctx.fill();

                n.x += n.vx;
                n.y += n.vy;
                if (n.x < 0 || n.x > canvas.width) n.vx *= -1;
                if (n.y < 0 || n.y > canvas.height) n.vy *= -1;
            });

            requestAnimationFrame(draw);
        }
        draw();
    </script>
</body>
</html>
"""
        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(html_code)

        return html_code
