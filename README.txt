HyperReason v2 — an honest Adaptive-Entropy MCTS test-time-compute engine.

v1.x of this project shipped an engine that never called a model and an invented
benchmark table. v2 is a full honest rebuild: it genuinely drives a real LLM
(Z.AI GLM by default; Ollama / HuggingFace supported), runs Adaptive-Entropy
MCTS over model-generated candidates, picks the answer by self-consistency, and
projects KV-cache savings from real per-node token counts.

Quickstart:
    pip install hyper-reason
    export ANTHROPIC_BASE_URL=https://api.z.ai/api/anthropic
    export ANTHROPIC_API_KEY=...
    python -c "from hyper_reason import wrap_model, GLMBackend; \
print(wrap_model(GLMBackend()).reason('What is 17 + 25?')['boxed_answer'])"

Real benchmark (GSM8K-mini, N=20, GLM-4.6):
    greedy (T=0)            : 95.0%
    self-consistency (K=4)  : 90.0%
    AE-MCTS (sims=6,k=2,d3) : 85.0%

We publish the run where AE-MCTS UNDERPERFORMS greedy: GLM-4.6 is already near
the ceiling on easy GSM8K, so modest-budget search does not pay off there. This
is the opposite of a SOTA claim. Re-run and scale it:
    python eval/gsm8k_mini.py --n 100 --sims 16 --k 4

Entropy is labeled "sample_diversity_entropy (no logprobs)". KV savings are a
projected simulator (no real GPU). Backends RAISE when offline — they never
fabricate a response. Raw per-problem traces: eval/runs/*.jsonl.

Full docs: https://github.com/rudra496/hyper-reason  (README.md)
Live demo: https://rudra496.github.io/hyper-reason
License: MIT. Author: Rudra Sarker.
