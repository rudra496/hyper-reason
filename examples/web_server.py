"""
Interactive Web Server & Local AI Playground for HyperReason Engine (v2)
Serves a lightweight live web GUI on http://127.0.0.1:8085 for interactive MCTS reasoning search.
"""

import http.server
import socketserver
import json
import os
import urllib.parse
from hyper_reason import ReasonEngine, SearchConfig, TreeVisualizer, GLMBackend, MockBackend

PORT = 8085

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>HyperReason — Local AI Reasoning Server</title>
    <style>
        body { background: #0a0c10; color: #f0f6fc; font-family: system-ui, sans-serif; padding: 2rem; }
        .container { max-width: 900px; margin: 0 auto; }
        h1 { color: #00f2fe; }
        textarea { width: 100%; height: 100px; background: #12161f; color: #fff; border: 1px solid #30363d; padding: 1rem; border-radius: 8px; font-size: 1rem; box-sizing: border-box; }
        button { background: #00f2fe; color: #000; font-weight: bold; border: none; padding: 12px 24px; border-radius: 6px; cursor: pointer; margin-top: 1rem; }
        button:hover { opacity: 0.9; }
        pre { background: #0d1117; padding: 1.5rem; border-radius: 8px; border: 1px solid #30363d; color: #4ade80; overflow-x: auto; font-family: monospace; }
    </style>
</head>
<body>
    <div class="container">
        <h1>⚡ HyperReason Local AI Engine Playground</h1>
        <p>Type a mathematical or logical word problem to execute Adaptive Entropy MCTS reasoning search:</p>
        <form method="POST" action="/solve">
            <textarea name="prompt">Janet has 3 boxes of apples. Each box contains 12 apples. She gives away 5 apples to her neighbor and eats 2. How many apples does she have left?</textarea><br>
            <button type="submit">Run AE-MCTS Reason Engine</button>
        </form>
        <h2>Output Trajectory & MCTS Tree Visualization:</h2>
        <pre>__OUTPUT__</pre>
    </div>
</body>
</html>
"""

class HyperReasonHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-type", "text/html; charset=utf-8")
            self.end_headers()
            html = HTML_PAGE.replace("__OUTPUT__", "Click 'Run AE-MCTS Reason Engine' to start...")
            self.wfile.write(html.encode("utf-8"))
        else:
            super().do_GET()

    def do_POST(self):
        if self.path == "/solve":
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length).decode('utf-8')
            parsed = urllib.parse.parse_qs(post_data)
            prompt = parsed.get("prompt", [""])[0]

            backend = GLMBackend() if os.environ.get("ANTHROPIC_API_KEY") else MockBackend()
            config = SearchConfig(num_simulations=12, max_depth=4, k_samples=2)
            engine = ReasonEngine(backend, config)

            res = engine.reason(prompt)

            visual_tree = ""
            if "root" in res and res["root"]:
                visualizer = TreeVisualizer(max_render_depth=3)
                visual_tree = visualizer.render(res["root"])

            out_text = f"PROBLEM PROMPT:\n{prompt}\n\n"
            if visual_tree:
                out_text += f"{visual_tree}\n\n"
            out_text += f"FINAL REASONING SOLUTION:\nBoxed Answer: {res.get('boxed_answer')}\nConfidence: {res.get('confidence')}\n\n"
            out_text += f"METRICS:\n{json.dumps(res.get('metrics', {}), indent=2)}\n"

            self.send_response(200)
            self.send_header("Content-type", "text/html; charset=utf-8")
            self.end_headers()
            html = HTML_PAGE.replace("__OUTPUT__", out_text)
            self.wfile.write(html.encode("utf-8"))

def start_server():
    print(f"🚀 Starting HyperReason Local Web Server on http://127.0.0.1:{PORT}...")
    with socketserver.TCPServer(("127.0.0.1", PORT), HyperReasonHTTPRequestHandler) as httpd:
        print(f"Serving live playground on http://127.0.0.1:{PORT}")
        httpd.serve_forever()

if __name__ == "__main__":
    start_server()
