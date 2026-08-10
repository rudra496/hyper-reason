"""
Interactive Web Server & Multi-Provider Playground for HyperReason Engine (v2)
Serves a sleek, modern, glassmorphism web GUI on http://127.0.0.1:8085.
Supports: Z.AI GLM, OpenAI / Compatible (DeepSeek, Groq, vLLM), Ollama, Transformers, and Mock.
"""

import http.server
import socketserver
import json
import os
import urllib.parse
from hyper_reason import (
    ReasonEngine,
    SearchConfig,
    TreeVisualizer,
    GLMBackend,
    OpenAIBackend,
    OllamaBackend,
    MockBackend,
)
try:
    from hyper_reason import TransformersBackend
except Exception:
    TransformersBackend = None

PORT = 8085

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>⚡ HyperReason — Multi-Provider AI Reasoning Engine</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg: #090d16;
            --card-bg: rgba(18, 26, 42, 0.75);
            --card-border: rgba(0, 242, 254, 0.15);
            --accent-cyan: #00f2fe;
            --accent-blue: #4facfe;
            --accent-purple: #9d4edd;
            --text-main: #f0f6fc;
            --text-muted: #8b949e;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            background-color: var(--bg);
            background-image: 
                radial-gradient(circle at 15% 15%, rgba(0, 242, 254, 0.08) 0%, transparent 40%),
                radial-gradient(circle at 85% 85%, rgba(157, 78, 221, 0.08) 0%, transparent 40%);
            color: var(--text-main);
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
            min-height: 100vh;
            padding: 2rem;
        }

        .container { max-width: 1100px; margin: 0 auto; }
        
        header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding-bottom: 1.5rem;
            margin-bottom: 2rem;
            border-bottom: 1px solid var(--card-border);
        }

        .logo { font-size: 1.6rem; font-weight: 700; background: linear-gradient(135deg, var(--accent-cyan), var(--accent-blue)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; display: flex; align-items: center; gap: 8px; }
        .tagline { color: var(--text-muted); font-size: 0.9rem; margin-top: 4px; }

        .badge-bar { display: flex; gap: 8px; }
        .badge { background: rgba(0, 242, 254, 0.1); color: var(--accent-cyan); border: 1px solid rgba(0, 242, 254, 0.3); padding: 4px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: 600; }
        .badge-live { background: rgba(74, 222, 128, 0.1); color: #4ade80; border-color: rgba(74, 222, 128, 0.3); }

        .grid-layout { display: grid; grid-template-columns: 340px 1fr; gap: 1.5rem; }

        .card {
            background: var(--card-bg);
            backdrop-filter: blur(12px);
            border: 1px solid var(--card-border);
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        }

        .form-group { margin-bottom: 1.2rem; }
        label { display: block; font-size: 0.85rem; font-weight: 600; color: var(--text-muted); margin-bottom: 6px; }
        
        select, input[type="text"], input[type="password"], textarea {
            width: 100%;
            background: rgba(10, 15, 25, 0.8);
            border: 1px solid rgba(255, 255, 255, 0.1);
            color: var(--text-main);
            padding: 10px 12px;
            border-radius: 8px;
            font-family: inherit;
            font-size: 0.9rem;
            transition: all 0.2s ease;
        }

        select:focus, input:focus, textarea:focus {
            outline: none;
            border-color: var(--accent-cyan);
            box-shadow: 0 0 0 2px rgba(0, 242, 254, 0.2);
        }

        textarea { height: 110px; resize: vertical; line-height: 1.5; }

        .slider-group { display: flex; align-items: center; justify-content: space-between; gap: 10px; }
        .slider-group input[type="range"] { flex: 1; accent-color: var(--accent-cyan); }
        .slider-value { font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; color: var(--accent-cyan); min-width: 24px; text-align: right; }

        .btn-submit {
            width: 100%;
            background: linear-gradient(135deg, var(--accent-cyan), var(--accent-blue));
            color: #000;
            font-weight: 700;
            font-size: 0.95rem;
            border: none;
            padding: 14px;
            border-radius: 8px;
            cursor: pointer;
            transition: transform 0.1s ease, box-shadow 0.2s ease;
            box-shadow: 0 4px 14px 0 rgba(0, 242, 254, 0.39);
        }
        .btn-submit:hover { transform: translateY(-1px); box-shadow: 0 6px 20px 0 rgba(0, 242, 254, 0.55); }
        .btn-submit:active { transform: translateY(0); }

        .output-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 1rem; }
        .output-header h2 { font-size: 1.1rem; font-weight: 600; }

        pre {
            background: #060911;
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 8px;
            padding: 1.25rem;
            color: #4ade80;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
            line-height: 1.5;
            overflow-x: auto;
            max-height: 600px;
        }

        .stats-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-bottom: 1.5rem; }
        .stat-box { background: rgba(10, 15, 25, 0.6); border: 1px solid rgba(255, 255, 255, 0.05); padding: 12px; border-radius: 8px; text-align: center; }
        .stat-val { font-family: 'JetBrains Mono', monospace; font-size: 1.2rem; font-weight: 700; color: var(--accent-cyan); }
        .stat-lbl { font-size: 0.75rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; margin-top: 4px; }

        @media (max-width: 850px) {
            .grid-layout { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div>
                <div class="logo">⚡ HyperReason Engine</div>
                <div class="tagline">Honest Adaptive-Entropy MCTS Test-Time Compute Playground</div>
            </div>
            <div class="badge-bar">
                <span class="badge">v2.0.1</span>
                <span class="badge badge-live">Multi-Backend</span>
            </div>
        </header>

        <form method="POST" action="/solve">
            <div class="grid-layout">
                <!-- Sidebar Configuration -->
                <div class="card">
                    <div class="form-group">
                        <label for="provider">LLM Provider</label>
                        <select id="provider" name="provider" onchange="updateProviderDefaults()">
                            <option value="mock" __SEL_MOCK__>Deterministic Mock (Offline / Free)</option>
                            <option value="glm" __SEL_GLM__>Z.AI / GLM (Anthropic API Compatible)</option>
                            <option value="openai" __SEL_OPENAI__>OpenAI / DeepSeek / Groq (Chat API)</option>
                            <option value="ollama" __SEL_OLLAMA__>Ollama (Local Offline LLM)</option>
                            <option value="transformers" __SEL_TRANSFORMERS__>HuggingFace Transformers (Local)</option>
                        </select>
                    </div>

                    <div class="form-group">
                        <label for="model">Model Identifier</label>
                        <input type="text" id="model" name="model" value="__VAL_MODEL__" placeholder="e.g. glm-4.6, gpt-4o-mini, llama3:8b">
                    </div>

                    <div class="form-group">
                        <label for="api_key">API Key (Optional override)</label>
                        <input type="password" id="api_key" name="api_key" value="__VAL_APIKEY__" placeholder="Auto-reads from ENV if empty">
                    </div>

                    <div class="form-group">
                        <label for="base_url">Custom Base URL (Optional)</label>
                        <input type="text" id="base_url" name="base_url" value="__VAL_BASEURL__" placeholder="e.g. https://api.z.ai/api/anthropic">
                    </div>

                    <hr style="border:0; border-top:1px solid rgba(255,255,255,0.08); margin: 1.5rem 0;">

                    <div class="form-group">
                        <label>MCTS Simulations: <span id="sims_val" class="slider-value">__VAL_SIMS__</span></label>
                        <div class="slider-group">
                            <input type="range" name="num_simulations" min="4" max="48" value="__VAL_SIMS__" oninput="document.getElementById('sims_val').innerText = this.value">
                        </div>
                    </div>

                    <div class="form-group">
                        <label>Max Search Depth: <span id="depth_val" class="slider-value">__VAL_DEPTH__</span></label>
                        <div class="slider-group">
                            <input type="range" name="max_depth" min="2" max="8" value="__VAL_DEPTH__" oninput="document.getElementById('depth_val').innerText = this.value">
                        </div>
                    </div>

                    <div class="form-group">
                        <label>K Candidates / Step: <span id="k_val" class="slider-value">__VAL_K__</span></label>
                        <div class="slider-group">
                            <input type="range" name="k_samples" min="1" max="4" value="__VAL_K__" oninput="document.getElementById('k_val').innerText = this.value">
                        </div>
                    </div>

                    <button type="submit" class="btn-submit">🚀 Run AE-MCTS Search</button>
                </div>

                <!-- Main Content Area -->
                <div>
                    <div class="card" style="margin-bottom: 1.5rem;">
                        <div class="form-group" style="margin-bottom: 0;">
                            <label for="prompt">Reasoning Prompt / Word Problem</label>
                            <textarea id="prompt" name="prompt">__VAL_PROMPT__</textarea>
                        </div>
                    </div>

                    <div class="card">
                        <div class="output-header">
                            <h2>📊 Reasoning Trajectory & MCTS Metrics</h2>
                        </div>

                        __STATS_HTML__

                        <pre>__OUTPUT__</pre>
                    </div>
                </div>
            </div>
        </form>
    </div>

    <script>
        function updateProviderDefaults() {
            const p = document.getElementById('provider').value;
            const m = document.getElementById('model');
            const url = document.getElementById('base_url');
            if (p === 'glm') {
                m.value = 'glm-4.6';
                url.value = 'https://api.z.ai/api/anthropic';
            } else if (p === 'openai') {
                m.value = 'gpt-4o-mini';
                url.value = 'https://api.openai.com/v1';
            } else if (p === 'ollama') {
                m.value = 'llama3:8b';
                url.value = 'http://localhost:11434';
            } else if (p === 'mock') {
                m.value = 'mock-heuristic';
                url.value = '';
            }
        }
    </script>
</body>
</html>
"""

class HyperReasonHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-type", "text/html; charset=utf-8")
            self.end_headers()

            html = HTML_PAGE
            html = html.replace("__SEL_MOCK__", "selected")
            html = html.replace("__SEL_GLM__", "")
            html = html.replace("__SEL_OPENAI__", "")
            html = html.replace("__SEL_OLLAMA__", "")
            html = html.replace("__SEL_TRANSFORMERS__", "")
            html = html.replace("__VAL_MODEL__", "mock-heuristic")
            html = html.replace("__VAL_APIKEY__", "")
            html = html.replace("__VAL_BASEURL__", "")
            html = html.replace("__VAL_SIMS__", "12")
            html = html.replace("__VAL_DEPTH__", "4")
            html = html.replace("__VAL_K__", "2")
            html = html.replace("__VAL_PROMPT__", "Janet has 3 boxes of apples. Each box contains 12 apples. She gives away 5 apples to her neighbor and eats 2. How many apples does she have left?")
            html = html.replace("__STATS_HTML__", "")
            html = html.replace("__OUTPUT__", "Select your provider on the left and click 'Run AE-MCTS Search' to execute reasoning...")
            
            self.wfile.write(html.encode("utf-8"))
        else:
            super().do_GET()

    def do_POST(self):
        if self.path == "/solve":
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length).decode('utf-8')
            parsed = urllib.parse.parse_qs(post_data)

            provider = parsed.get("provider", ["mock"])[0]
            model_name = parsed.get("model", [""])[0]
            api_key = parsed.get("api_key", [""])[0] or None
            base_url = parsed.get("base_url", [""])[0] or None
            prompt = parsed.get("prompt", [""])[0]

            num_sims = int(parsed.get("num_simulations", [12])[0])
            max_depth = int(parsed.get("max_depth", [4])[0])
            k_samples = int(parsed.get("k_samples", [2])[0])

            # Select Backend dynamically
            backend = None
            try:
                if provider == "glm":
                    backend = GLMBackend(model=model_name or "glm-4.6", api_key=api_key, base_url=base_url)
                elif provider == "openai":
                    backend = OpenAIBackend(model=model_name or "gpt-4o-mini", api_key=api_key, base_url=base_url)
                elif provider == "ollama":
                    backend = OllamaBackend(model=model_name or "llama3:8b", host=base_url or "http://localhost:11434")
                elif provider == "transformers" and TransformersBackend:
                    backend = TransformersBackend(model=model_name or "gpt2")
                else:
                    backend = MockBackend()
            except Exception as e:
                backend = MockBackend()

            config = SearchConfig(num_simulations=num_sims, max_depth=max_depth, k_samples=k_samples)
            engine = ReasonEngine(backend, config)

            try:
                res = engine.reason(prompt)
                metrics = res.get("metrics", {})
                boxed_answer = res.get("boxed_answer", "N/A")
                confidence = res.get("confidence", 0.0)

                visual_tree = ""
                if "root" in res and res["root"]:
                    visualizer = TreeVisualizer(max_render_depth=3)
                    visual_tree = visualizer.render(res["root"])

                stats_html = f"""
                <div class="stats-grid">
                    <div class="stat-box">
                        <div class="stat-val">{boxed_answer}</div>
                        <div class="stat-lbl">Boxed Answer</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-val">{confidence}</div>
                        <div class="stat-lbl">Confidence Score</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-val">{metrics.get('simulations_executed', 0)}</div>
                        <div class="stat-lbl">MCTS Expansions</div>
                    </div>
                </div>
                """

                out_text = f"PROBLEM PROMPT:\n{prompt}\n\n"
                if visual_tree:
                    out_text += f"{visual_tree}\n\n"
                out_text += f"FINAL REASONING SOLUTION:\nBoxed Answer: {boxed_answer}\nConfidence: {confidence}\n\n"
                out_text += f"FULL METRICS:\n{json.dumps(metrics, indent=2)}\n"

            except Exception as ex:
                stats_html = ""
                out_text = f"ERROR EXECUTING REASONING SEARCH:\n{str(ex)}"

            html = HTML_PAGE
            html = html.replace(f"__SEL_{provider.upper()}__", "selected")
            for p in ["MOCK", "GLM", "OPENAI", "OLLAMA", "TRANSFORMERS"]:
                html = html.replace(f"__SEL_{p}__", "")
            
            html = html.replace("__VAL_MODEL__", model_name)
            html = html.replace("__VAL_APIKEY__", api_key or "")
            html = html.replace("__VAL_BASEURL__", base_url or "")
            html = html.replace("__VAL_SIMS__", str(num_sims))
            html = html.replace("__VAL_DEPTH__", str(max_depth))
            html = html.replace("__VAL_K__", str(k_samples))
            html = html.replace("__VAL_PROMPT__", prompt)
            html = html.replace("__STATS_HTML__", stats_html)
            html = html.replace("__OUTPUT__", out_text)

            self.send_response(200)
            self.send_header("Content-type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(html.encode("utf-8"))

def start_server():
    print(f"🚀 Starting HyperReason Multi-Provider Server on http://127.0.0.1:{PORT}...")
    with socketserver.TCPServer(("127.0.0.1", PORT), HyperReasonHTTPRequestHandler) as httpd:
        print(f"Serving live playground on http://127.0.0.1:{PORT}")
        httpd.serve_forever()

if __name__ == "__main__":
    start_server()
