HyperReason v2 (hypermcts) — an honest Adaptive-Entropy MCTS test-time-compute engine.

Supports all model backends: Z.AI GLM, OpenAI / DeepSeek / Groq, Ollama, Transformers, and Mock.

Quickstart:
    pip install hypermcts
    export ANTHROPIC_BASE_URL=https://api.z.ai/api/anthropic
    export ANTHROPIC_API_KEY=...
    python -c "from hyper_reason import wrap_model, GLMBackend; \
print(wrap_model(GLMBackend()).reason('What is 17 + 25?')['boxed_answer'])"

Run Interactive Web Playground:
    python examples/web_server.py
    Open http://127.0.0.1:8088 in browser

Real benchmark (GSM8K-mini, N=20, GLM-4.6):
    greedy (T=0)            : 95.0%
    self-consistency (K=4)  : 90.0%
    AE-MCTS (sims=6,k=2,d3) : 85.0%

We publish the run where AE-MCTS UNDERPERFORMS greedy: GLM-4.6 is already near
the ceiling on easy GSM8K, so modest-budget search does not pay off there.

Full docs: https://github.com/rudra496/hyper-reason  (README.md)
Live demo: https://rudra496.github.io/hyper-reason
Author: Rudra Sarker (https://rudra496.github.io/site)
GitHub: https://github.com/rudra496
LinkedIn: https://linkedin.com/in/rudrasarker
X/Twitter: https://x.com/Rudra496
License: MIT.
